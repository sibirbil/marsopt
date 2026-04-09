# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
import numpy as np
cimport numpy as cnp
from cpython.pycapsule cimport PyCapsule_GetPointer
from time import perf_counter
from libc.math cimport exp as cexp, log as clog, cos as ccos, sqrt as csqrt, fabs, round as cround, M_PI
from libc.string cimport memset
from libcpp.vector cimport vector

from marsopt.variable cimport Variable, CategoryIndexer
from marsopt.logger import OptimizationLogger

from functools import lru_cache

cnp.import_array()

# ============================================================================
# BitGenerator C API (PCG64)
# ============================================================================

ctypedef struct bitgen_t:
    void *state
    cnp.uint64_t (*next_uint64)(void *state) nogil
    cnp.uint32_t (*next_uint32)(void *state) nogil
    double (*next_double)(void *state) nogil
    cnp.uint64_t (*next_raw)(void *state) nogil


# Matches rk_double() exactly: uses next_double from MT19937 capsule
cdef inline double _rng_double(bitgen_t *bg) noexcept nogil:
    return bg.next_double(bg.state)


# Matches RandomState.uniform(low, high) for scalar
cdef inline double _rng_uniform(bitgen_t *bg, double low, double high) noexcept nogil:
    return low + (high - low) * bg.next_double(bg.state)


# Matches rk_gauss() exactly: Marsaglia polar method with gauss caching
cdef inline double _rng_gauss(bitgen_t *bg, bint *has_gauss, double *gauss_cache) noexcept nogil:
    cdef double f, x1, x2, r2
    if has_gauss[0]:
        has_gauss[0] = 0
        return gauss_cache[0]
    while True:
        x1 = 2.0 * bg.next_double(bg.state) - 1.0
        x2 = 2.0 * bg.next_double(bg.state) - 1.0
        r2 = x1 * x1 + x2 * x2
        if r2 < 1.0 and r2 != 0.0:
            break
    f = csqrt(-2.0 * clog(r2) / r2)
    gauss_cache[0] = f * x1
    has_gauss[0] = 1
    return f * x2


cdef inline double _rng_normal(bitgen_t *bg, double mu, double sigma,
                                bint *has_gauss, double *gauss_cache) noexcept nogil:
    return mu + sigma * _rng_gauss(bg, has_gauss, gauss_cache)


# Matches rk_interval() exactly: rejection sampling with next_uint32
cdef inline long _rng_interval(bitgen_t *bg, cnp.uint32_t max_val) noexcept nogil:
    cdef cnp.uint32_t mask = max_val
    cdef cnp.uint32_t value
    if max_val == 0:
        return 0
    mask |= mask >> 1
    mask |= mask >> 2
    mask |= mask >> 4
    mask |= mask >> 8
    mask |= mask >> 16
    while True:
        value = bg.next_uint32(bg.state) & mask
        if value <= max_val:
            break
    return <long>value


# Matches RandomState.randint(0, n) for scalar
cdef inline int _rng_randint(bitgen_t *bg, int n) noexcept nogil:
    if n <= 1:
        return 0
    return <int>_rng_interval(bg, <cnp.uint32_t>(n - 1))


# Matches RandomState.choice(n, p=probs): CDF inversion with next_double
cdef inline int _rng_choice_p(bitgen_t *bg, int n, double *probs) noexcept nogil:
    cdef double r = bg.next_double(bg.state)
    cdef double cumsum = 0.0
    cdef int i
    for i in range(n - 1):
        cumsum += probs[i]
        if r < cumsum:
            return i
    return n - 1


# ============================================================================
# C-level helper functions
# ============================================================================

cdef inline double _reflect_at_boundaries(double x, double low, double high) noexcept nogil:
    while True:
        if x < low:
            x = low + (low - x) * 0.5
        elif x > high:
            x = high - (x - high) * 0.5
        else:
            break
    return x


cdef void _build_hist_and_smooth(
    double *elite_var, int n_elites, int low, int n_values,
    double sigma_t, double *scores_out, double *hist_sum_out
) noexcept nogil:
    cdef double hist[20]
    cdef int i, j, k
    cdef double v, diff, kernel_sum, kernel_val

    memset(hist, 0, 20 * sizeof(double))
    memset(scores_out, 0, n_values * sizeof(double))

    hist_sum_out[0] = 0.0
    for i in range(n_elites):
        v = elite_var[i]
        if v != v:
            continue
        k = <int>v - low
        if 0 <= k < n_values:
            hist[k] += 1.0
            hist_sum_out[0] += 1.0

    if hist_sum_out[0] == 0.0:
        return

    for j in range(n_values):
        if hist[j] == 0.0:
            continue
        kernel_sum = 0.0
        for k in range(n_values):
            diff = <double>(k - j) / sigma_t
            kernel_val = cexp(-0.5 * diff * diff)
            kernel_sum += kernel_val
        for k in range(n_values):
            diff = <double>(k - j) / sigma_t
            kernel_val = cexp(-0.5 * diff * diff)
            scores_out[k] += hist[j] * kernel_val / kernel_sum


cdef inline void _normalize_scores(double *scores, int n, double noise) noexcept nogil:
    cdef double floor_val = (1.0 / n) * noise
    cdef double s = 0.0
    cdef int i
    for i in range(n):
        s += scores[i]
    for i in range(n):
        scores[i] = (1.0 - floor_val) * (scores[i] / s) + floor_val / n


cdef void _reflect_categorical_noise(double *noisy, int size) noexcept nogil:
    cdef int i
    cdef bint any_oob = True
    while any_oob:
        any_oob = False
        for i in range(size):
            if noisy[i] < 0.0:
                noisy[i] = -noisy[i] * 0.5
                any_oob = True
            elif noisy[i] > 1.0:
                noisy[i] = 1.0 - (noisy[i] - 1.0) * 0.5
                any_oob = True


cdef void _softmax_inplace(double *arr, int size, double temp) noexcept nogil:
    cdef int i
    cdef double max_val = arr[0]
    cdef double sum_exp = 0.0
    for i in range(1, size):
        if arr[i] > max_val:
            max_val = arr[i]
    for i in range(size):
        arr[i] = cexp((arr[i] - max_val) * temp)
        sum_exp += arr[i]
    for i in range(size):
        arr[i] /= sum_exp


cdef int _collect_elite_inbounds(
    double *var_values, long *elite_idx, int n_elites,
    double low, double high, vector[double] &out_buf
) noexcept nogil:
    cdef int i
    cdef double v
    out_buf.clear()
    for i in range(n_elites):
        v = var_values[elite_idx[i]]
        if v >= low and v <= high and v == v:
            out_buf.push_back(v)
    return <int>out_buf.size()


cdef void _compute_cat_freq(
    int *var_values, long *elite_idx, int n_elites,
    int *cat_indices, int cat_size,
    double *freq_out, double *total_out,
    vector[int] &counts_buf
) noexcept nogil:
    cdef int i, j, val, max_cat_idx

    max_cat_idx = 0
    for i in range(cat_size):
        if cat_indices[i] > max_cat_idx:
            max_cat_idx = cat_indices[i]

    counts_buf.assign(max_cat_idx + 1, 0)
    memset(freq_out, 0, cat_size * sizeof(double))

    total_out[0] = 0.0
    for i in range(n_elites):
        val = var_values[elite_idx[i]]
        if val >= 0 and val <= max_cat_idx:
            counts_buf[val] += 1

    for j in range(cat_size):
        freq_out[j] = <double>counts_buf[cat_indices[j]]
        total_out[0] += freq_out[j]


# ============================================================================
# Trial class
# ============================================================================

class Trial:
    """Represents a single trial in the optimization process."""

    __slots__ = [
        "study",
        "trial_id",
        "variables",
        "_validated_variables",
        "user_attrs",
    ]

    def __init__(self, study, int trial_id):
        self.study = study
        self.trial_id = trial_id
        self.variables = {}
        self._validated_variables = set()
        self.user_attrs = {}

    def add_attr(self, str name, value):
        """
        Add a user-defined attribute to the trial.

        Parameters
        ----------
        name : str
            The name of the attribute.
        value : Any
            The value of the attribute.
        """
        self.user_attrs[name] = value

    def __repr__(self):
        return f"Trial(trial_id={self.trial_id}, variables={self.variables}, user_attrs={self.user_attrs})"

    @staticmethod
    @lru_cache(maxsize=None)
    def _validate_numerical_cached(name, low, high, expected_type, log):
        if expected_type is float:
            if not (isinstance(low, (int, float)) and isinstance(high, (int, float))):
                raise TypeError(
                    f"Variable '{name}': 'low' and 'high' must be numeric, got {type(low)} and {type(high)}"
                )
        elif expected_type is int:
            if not (isinstance(low, int) and isinstance(high, int)):
                raise TypeError(
                    f"Variable '{name}': 'low' and 'high' must be integers, got {type(low)} and {type(high)}"
                )
        else:
            raise TypeError(f"Variable '{name}': Unsupported type {expected_type}")
        low = expected_type(low)
        high = expected_type(high)
        if low >= high:
            raise ValueError(
                f"Variable '{name}': 'low' must be less than 'high' (got {low} >= {high})"
            )
        if log and (low <= 0 or high <= 0):
            raise ValueError(
                f"Variable '{name}': 'low' and 'high' must be positive when 'log' is True (got {low}, {high})"
            )

    def _validate_numerical(self, name, low, high, expected_type, log):
        if not isinstance(name, str):
            raise TypeError(f"Variable name must be a string, got {type(name)}")
        if name == "":
            raise ValueError("Variable name cannot be an empty string.")
        Trial._validate_numerical_cached(name, low, high, expected_type, log)
        self._validated_variables.add(name)

    @staticmethod
    @lru_cache(maxsize=None)
    def _validate_categorical_cached(name, categories_tuple):
        if len(categories_tuple) < 1:
            raise ValueError(
                f"Variable '{name}': 'categories' must contain at least one element"
            )
        if len(set(categories_tuple)) != len(categories_tuple):
            raise ValueError(
                f"Variable '{name}': 'categories' contains duplicate values"
            )
        try:
            _ = categories_tuple[0]
        except (TypeError, IndexError):
            raise TypeError(
                f"Variable '{name}': 'categories' must be indexable, got {type(categories_tuple)} with non-indexable elements"
            )

    def _validate_categorical(self, name, categories):
        if not isinstance(name, str):
            raise TypeError(f"Variable name must be a string, got {type(name)}")
        if name == "":
            raise ValueError("Variable name cannot be an empty string.")
        if not isinstance(categories, list):
            raise TypeError(
                f"Variable '{name}': 'categories' must be a list, got {type(categories)}"
            )
        if any(not isinstance(cat, str) for cat in categories):
            raise TypeError(
                f"Variable '{name}': all items in 'categories' must be strings."
            )
        categories_tuple = tuple(categories)
        Trial._validate_categorical_cached(name, categories_tuple)
        self._validated_variables.add(name)

    def suggest_float(self, str name, double low, double high, bint log=False):
        """
        Suggest a floating-point variable value.

        Parameters
        ----------
        name : str
            The name of the variable.
        low : float
            The lower bound of the variable range.
        high : float
            The upper bound of the variable range.
        log : bool, optional, default = False
            Whether the variable is log-scaled.

        Returns
        -------
        float
            The suggested floating-point value.
        """
        self._validate_numerical(name, low, high, float, log)
        value = self.study._suggest_numerical(name, low, high, float, log)
        self.variables[name] = value
        return value

    def suggest_int(self, str name, int low, int high, bint log=False):
        """
        Suggest an integer variable value.

        Parameters
        ----------
        name : str
            The name of the variable.
        low : int
            The lower bound of the variable range.
        high : int
            The upper bound of the variable range.
        log : bool, optional, default = False
            Whether the variable is log-scaled.

        Returns
        -------
        int
            The suggested integer value.
        """
        self._validate_numerical(name, low, high, int, log)
        value = int(self.study._suggest_numerical(name, low, high, int, log))
        self.variables[name] = value
        return value

    def suggest_categorical(self, str name, list categories):
        """
        Suggest a categorical variable value.

        Parameters
        ----------
        name : str
            The name of the variable.
        categories : List[str]
            A list of valid string categorical values.

        Returns
        -------
        str
            The suggested categorical string value.
        """
        self._validate_categorical(name, categories)
        value = self.study._suggest_categorical(name, categories)
        self.variables[name] = value
        return value


# ============================================================================
# Study class
# ============================================================================

cdef class Study:
    """Mixed Adaptive Random Search for Optimization"""

    cdef public int n_trials
    cdef public object n_init_points
    cdef public double initial_noise
    cdef public object final_noise
    cdef public bint verbose
    cdef public str direction
    cdef public double epsilon
    cdef public object elite_window

    cdef object _rng_obj        # numpy Generator (Python-level access)
    cdef object _rng_bg_obj     # PCG64 BitGenerator (prevent GC)
    cdef bitgen_t *_bg          # C-level pointer into PCG64
    cdef bint _has_gauss        # gauss cache for C-level normal()
    cdef double _gauss_cache
    cdef cnp.ndarray _objective_values_arr
    cdef cnp.ndarray _elapsed_times_arr
    cdef object _current_trial
    cdef list _trials
    cdef dict _variables

    cdef double _progress
    cdef double _current_noise
    cdef int _current_n_elites
    cdef double _current_cat_temp
    cdef double _direction_multiplier
    cdef cnp.ndarray _elite_indices_arr
    cdef bint _force_random
    cdef dict _evo_path
    cdef object _best_ever_idx
    cdef object _logger

    cdef double _scores_buf[20]
    cdef vector[double] _elite_buf
    cdef vector[double] _cat_freq_buf
    cdef vector[int] _cat_counts_buf

    # Incremental sorted elite tracking (replaces argpartition)
    cdef vector[double] _sorted_obj     # objective values * direction_multiplier, ascending
    cdef vector[long] _sorted_idx       # corresponding trial indices

    def __init__(
        self,
        double initial_noise=0.33,
        str direction="minimize",
        n_init_points=None,
        final_noise=None,
        double epsilon=1.0,
        elite_window=None,
        random_state=None,
        verbose=True,
    ):
        """
        Initialize the Study.

        Parameters
        ----------
        initial_noise : float, default = 0.33
            Initial noise level.
        direction : str, default = "minimize"
            Direction of optimization, either "minimize" or "maximize".
        n_init_points : int, default = None
            Number of initial random points. If ``None``, it is set as:
            ``round(sqrt(n_trials))``
        final_noise : float, default = None
            Final noise level. If ``None``, it is set as:
            ``min(1.0 / n_trials, initial_noise)``
        epsilon : float, default = 1.0
            Epsilon-greedy exploration constant. At each adaptive trial, with
            probability ``epsilon / (t + 1)`` a uniform random sample is drawn
            instead of the elite-guided step (harmonic decay).
        elite_window : int, default = None
            If set, only the most recent ``elite_window`` completed trials are
            considered for elite selection. If ``None``, full history is used.
        random_state : int, default = None
            Seed for reproducibility. Uses PCG64 BitGenerator internally.
            If ``None``, a random SeedSequence is used.
        verbose : bool, default = True
            Whether to print logs during optimization.
        """
        self._validate_init_params(
            n_init_points=n_init_points,
            random_state=random_state,
            final_noise=final_noise,
            initial_noise=initial_noise,
            direction=direction,
            verbose=verbose,
            epsilon=epsilon,
            elite_window=elite_window,
        )

        self.n_init_points = n_init_points
        self.initial_noise = initial_noise
        self.verbose = verbose
        self.direction = direction
        self.final_noise = final_noise
        self.epsilon = epsilon
        self.elite_window = elite_window

        # PCG64 C-level RNG via BitGenerator capsule API
        self._rng_bg_obj = np.random.PCG64(random_state if random_state is not None else np.random.SeedSequence())
        self._rng_obj = np.random.Generator(self._rng_bg_obj)
        cdef object capsule = self._rng_bg_obj.capsule
        self._bg = <bitgen_t*>PyCapsule_GetPointer(capsule, "BitGenerator")
        self._has_gauss = 0
        self._gauss_cache = 0.0

        self._objective_values_arr = None
        self._elapsed_times_arr = None
        self._current_trial = None
        self._trials = []
        self._variables = {}

        self._progress = 0.0
        self._current_noise = 0.0
        self._current_n_elites = 0
        self._current_cat_temp = 0.0
        self._direction_multiplier = 0.0
        self._elite_indices_arr = None
        self._force_random = False
        self._evo_path = {}
        self._best_ever_idx = None
        self._logger = OptimizationLogger() if verbose else None

    def __repr__(self):
        return (
            f"Study(n_init_points={self.n_init_points}, "
            f"initial_noise={self.initial_noise}, "
            f"final_noise={self.final_noise}, "
            f"direction='{self.direction}', "
            f"verbose={self.verbose})"
        )

    @property
    def _rng(self):
        return self._rng_obj

    @property
    def _objective_values(self):
        return self._objective_values_arr

    @_objective_values.setter
    def _objective_values(self, value):
        self._objective_values_arr = value

    @property
    def _elapsed_times(self):
        return self._elapsed_times_arr

    @_elapsed_times.setter
    def _elapsed_times(self, value):
        self._elapsed_times_arr = value

    @property
    def _elite_indices(self):
        return self._elite_indices_arr

    @_elite_indices.setter
    def _elite_indices(self, value):
        self._elite_indices_arr = value

    def _suggest_numerical(self, str name, low, high, var_type, bint log):
        cdef Variable var
        cdef int trial_id, n_values
        cdef bint small_int, medium_int
        cdef double value, sigma_t, base_value, drift
        cdef double var_range, noise_val, log_base, log_high_val, log_low_val, log_range
        cdef double hist_total, v_tmp
        cdef int i, n_elites_count, n_elites_local, n_valid, rand_idx
        cdef cnp.ndarray[cnp.float64_t, ndim=1] elite_var
        cdef double *var_data
        cdef long *elite_data
        cdef double *obj_data
        cdef double oh_buf[20]
        cdef double *bv
        cdef double *bo
        cdef bitgen_t *bg = self._bg

        py_var = self._variables.get(name)

        if py_var is None:
            py_var = Variable(name=name)
            py_var.set_values(max_iter=self.n_trials, var_type_or_categories=var_type)
            self._variables[name] = py_var
        else:
            if py_var.type != var_type:
                raise TypeError(
                    f"Variable '{name}' has already been registered with type {py_var.type}, "
                    f"but an attempt was made to register it as type {var_type}. Ensure consistency."
                )

        var = <Variable>py_var
        trial_id = self._current_trial.trial_id
        n_values = (high - low + 1) if var_type is int else 0
        small_int = var_type is int and not log and n_values <= 20
        medium_int = False

        if trial_id < self.n_init_points or self._force_random:
            if small_int or medium_int:
                value = low + _rng_randint(bg, n_values)
            else:
                if log:
                    value = cexp(_rng_uniform(bg, clog(<double>low), clog(<double>high)))
                else:
                    value = _rng_uniform(bg, <double>low, <double>high)

        elif small_int:
            elite_var = var.values[self._elite_indices_arr]
            n_elites_count = elite_var.shape[0]

            _build_hist_and_smooth(
                <double*>cnp.PyArray_DATA(elite_var), n_elites_count,
                <int>low, n_values, 0.35 + 0.65 * (1.0 - self._progress),
                self._scores_buf, &hist_total
            )

            if hist_total == 0.0:
                value = low + _rng_randint(bg, n_values)
            else:
                _normalize_scores(self._scores_buf, n_values, self._current_noise)
                value = low + _rng_choice_p(bg, n_values, self._scores_buf)

        elif medium_int:
            elite_var = var.values[self._elite_indices_arr]
            self._elite_buf.clear()
            for i in range(elite_var.shape[0]):
                v_tmp = (<double*>cnp.PyArray_DATA(elite_var))[i]
                if v_tmp == v_tmp and low <= v_tmp <= high:
                    self._elite_buf.push_back(v_tmp)
            n_valid = <int>self._elite_buf.size()

            if n_valid == 0:
                value = low + _rng_randint(bg, n_values)
            else:
                rand_idx = _rng_randint(bg, n_valid)
                base_value = self._elite_buf[rand_idx]
                sigma_t = self._current_noise * csqrt(<double>n_values)
                value = base_value + <int>cround(_rng_normal(bg, 0.0, sigma_t, &self._has_gauss, &self._gauss_cache))
                while value < low or value > high:
                    if value < low:
                        value = low + (low - value)
                    if value > high:
                        value = high - (value - high)

        else:
            var_data = <double*>cnp.PyArray_DATA(var.values)
            elite_data = <long*>cnp.PyArray_DATA(self._elite_indices_arr)
            n_elites_count = self._elite_indices_arr.shape[0]

            n_valid = _collect_elite_inbounds(
                var_data, elite_data, n_elites_count,
                <double>low, <double>high, self._elite_buf
            )

            if n_valid > 0:
                rand_idx = _rng_randint(bg, n_valid)
                base_value = self._elite_buf[rand_idx]
            else:
                self._elite_buf.clear()
                for i in range(trial_id):
                    v_tmp = var_data[i]
                    if v_tmp >= low and v_tmp <= high and v_tmp == v_tmp:
                        self._elite_buf.push_back(v_tmp)
                n_valid = <int>self._elite_buf.size()

                if n_valid == 0:
                    if log:
                        value = cexp(_rng_uniform(bg, clog(<double>low), clog(<double>high)))
                    else:
                        value = _rng_uniform(bg, <double>low, <double>high)
                    if var_type is int:
                        value = <int>value + <int>(
                            (_rng_double(bg) < fabs(value - <int>value))
                            * (1 if value > 0 else -1)
                        )
                    var.values[trial_id] = value
                    return value

                n_elites_local = min(self._current_n_elites, n_valid)
                if n_elites_local >= n_valid:
                    rand_idx = _rng_randint(bg, n_valid)
                    base_value = self._elite_buf[rand_idx]
                else:
                    obj_data = <double*>cnp.PyArray_DATA(self._objective_values_arr)
                    buf_vals = np.empty(n_valid, dtype=np.float64)
                    buf_objs = np.empty(n_valid, dtype=np.float64)
                    bv = <double*>cnp.PyArray_DATA(buf_vals)
                    bo = <double*>cnp.PyArray_DATA(buf_objs)
                    n_valid = 0
                    for i in range(trial_id):
                        v_tmp = var_data[i]
                        if v_tmp >= low and v_tmp <= high and v_tmp == v_tmp:
                            bv[n_valid] = v_tmp
                            bo[n_valid] = self._direction_multiplier * obj_data[i]
                            n_valid += 1
                    top_k = np.argpartition(buf_objs[:n_valid], n_elites_local - 1)[:n_elites_local]
                    rand_idx = _rng_randint(bg, n_elites_local)
                    base_value = bv[<int>top_k[rand_idx]]

            drift = 0.1 * self._evo_path.get(name, 0.0) * (1.0 - self._progress)

            if log:
                log_base = clog(base_value)
                log_high_val = clog(<double>high)
                log_low_val = clog(<double>low)
                log_range = log_high_val - log_low_val
                noise_val = _rng_normal(bg, 0.0, self._current_noise * log_range, &self._has_gauss, &self._gauss_cache)
                value = cexp(
                    _reflect_at_boundaries(log_base + noise_val + drift, log_low_val, log_high_val)
                )
            else:
                var_range = <double>high - <double>low
                noise_val = _rng_normal(bg, 0.0, self._current_noise * var_range, &self._has_gauss, &self._gauss_cache)
                value = _reflect_at_boundaries(base_value + noise_val + drift, <double>low, <double>high)

        if var_type is int and not small_int and not medium_int:
            value = <int>value + <int>((_rng_double(bg) < fabs(value - <int>value)) * (1 if value > 0 else -1))

        var.values[self._current_trial.trial_id] = value
        return value

    def _suggest_categorical(self, str name, list categories):
        cdef Variable var
        cdef int trial_id, cat_size, category_idx
        cdef double total, dominance, div_factor, p, explore_prob, temp
        cdef cnp.ndarray[cnp.int32_t, ndim=1] cat_indices
        cdef vector[double] noisy_buf
        cdef int n_elites_count
        cdef int *var_int_data
        cdef long *elite_data
        cdef int i
        cdef bitgen_t *bg = self._bg

        py_var = self._variables.get(name)
        trial_id = self._current_trial.trial_id

        if py_var is None:
            py_var = Variable(name=name)
            py_var.set_values(max_iter=self.n_trials, var_type_or_categories=categories)
            self._variables[name] = py_var
        else:
            if py_var.type is not type(categories):
                raise TypeError(
                    f"Variable '{name}' has already been registered with type {py_var.type}, "
                    f"but an attempt was made to register it as type {type(categories)}. Ensure consistency."
                )

        var = <Variable>py_var
        cat_indices = var.category_indexer.get_indices(categories)
        cat_size = cat_indices.shape[0]

        if trial_id < self.n_init_points or self._force_random:
            category_idx = cat_indices[_rng_randint(bg, cat_size)]
        else:
            var_int_data = <int*>cnp.PyArray_DATA(var.values)
            elite_data = <long*>cnp.PyArray_DATA(self._elite_indices_arr)
            n_elites_count = self._elite_indices_arr.shape[0]

            self._cat_freq_buf.resize(cat_size)
            _compute_cat_freq(
                var_int_data, elite_data, n_elites_count,
                <int*>cnp.PyArray_DATA(cat_indices), cat_size,
                self._cat_freq_buf.data(), &total,
                self._cat_counts_buf
            )

            if total > 0.0:
                dominance = 0.0
                for i in range(cat_size):
                    if self._cat_freq_buf[i] > dominance:
                        dominance = self._cat_freq_buf[i]
                dominance = dominance / total
            else:
                dominance = 0.5
            div_factor = max(0.0, (dominance - 1.0 / cat_size)) / (1.0 - 1.0 / cat_size + 1e-10)
            p = self._progress
            explore_prob = (1.0 / cat_size) * div_factor * 4.0 * p * (1.0 - p)

            if _rng_double(bg) < explore_prob:
                category_idx = cat_indices[_rng_randint(bg, cat_size)]
            else:
                if total == 0.0:
                    category_idx = cat_indices[_rng_randint(bg, cat_size)]
                    var.values[trial_id] = category_idx
                    return var.category_indexer.get_strings(category_idx)

                noisy_buf.resize(cat_size)
                for i in range(cat_size):
                    noisy_buf[i] = self._cat_freq_buf[i] / total + _rng_normal(bg, 0.0, self._current_noise, &self._has_gauss, &self._gauss_cache)

                _reflect_categorical_noise(noisy_buf.data(), cat_size)

                temp = self._current_cat_temp if self._current_cat_temp != 0.0 else 1.0
                _softmax_inplace(noisy_buf.data(), cat_size, temp)

                category_idx = cat_indices[_rng_choice_p(bg, cat_size, noisy_buf.data())]

        var.values[trial_id] = category_idx
        return var.category_indexer.get_strings(category_idx)

    @staticmethod
    def _reflect_at_boundaries(double x, double low=0.0, double high=1.0):
        return _reflect_at_boundaries(x, low, high)

    def _sample_value(self, double low, double high, bint log):
        cdef bitgen_t *bg = self._bg
        if log:
            return cexp(_rng_uniform(bg, clog(low), clog(high)))
        else:
            return _rng_uniform(bg, low, high)

    def optimize(self, objective_function, int n_trials):
        """
        Runs the optimization loop.

        Parameters
        ----------
        objective_function : Callable[[Trial], Union[float, int]]
            The function to optimize.
        n_trials : int
            The number of trials.
        """
        cdef int iteration, n_exist_trials, total_trials, pool_size, k, window_start
        cdef double best_value, obj_value, start_time
        cdef double elite_scale, noise_range, cos_anneal, eps_t
        cdef double progress, final_noise_val
        cdef bint is_new_best
        cdef double old_val, new_val, prev
        cdef double *obj_ptr = NULL
        cdef double *time_ptr = NULL
        cdef bitgen_t *bg = self._bg
        cdef double scaled_obj
        cdef int lo, hi, mid
        cdef int sorted_size, i, n_valid

        if not isinstance(n_trials, int):
            raise TypeError("n_trials must be an integer.")
        if n_trials <= 0:
            raise ValueError("n_trials must be a positive integer.")
        if not callable(objective_function):
            raise TypeError("objective_function must be a callable function.")

        if self._trials:
            n_exist_trials = len(self._trials)

            if self.direction == "minimize":
                best_iteration = int(np.argmin(self._objective_values_arr[:n_exist_trials]))
                best_value = self._objective_values_arr[best_iteration]
            else:
                best_iteration = int(np.argmax(self._objective_values_arr[:n_exist_trials]))
                best_value = self._objective_values_arr[best_iteration]

            total_trials = n_trials + n_exist_trials
            self.n_trials = total_trials

            old_obj = self._objective_values_arr
            old_times = self._elapsed_times_arr

            self._objective_values_arr = np.empty(total_trials, dtype=np.float64)
            self._elapsed_times_arr = np.empty(total_trials, dtype=np.float64)

            self._objective_values_arr[:n_exist_trials] = old_obj[:n_exist_trials]
            self._elapsed_times_arr[:n_exist_trials] = old_times[:n_exist_trials]

            for var_key in self._variables:
                self._variables[var_key].add_iter(n_trials)
        else:
            if self.verbose:
                self._logger.log_start(n_trials)

            n_exist_trials = 0
            total_trials = n_trials

            if self.direction == "minimize":
                best_value = float("inf")
            else:
                best_value = float("-inf")

            best_iteration = None

            if self.final_noise is None:
                self.final_noise = max(1e-7, min(1.0 / n_trials, self.initial_noise))

            self.n_trials = n_trials
            self._objective_values_arr = np.empty(n_trials, dtype=np.float64)
            self._elapsed_times_arr = np.empty(n_trials, dtype=np.float64)

            if self.n_init_points is None:
                self.n_init_points = max(10, round(csqrt(<double>self.n_trials)))

        elite_scale = 2.0 * csqrt(<double>total_trials)
        self._direction_multiplier = 1.0 if self.direction == "minimize" else -1.0

        obj_ptr = <double*>cnp.PyArray_DATA(self._objective_values_arr)
        time_ptr = <double*>cnp.PyArray_DATA(self._elapsed_times_arr)

        # Rebuild sorted elite list for resume case
        if n_exist_trials > 0:
            self._sorted_obj.clear()
            self._sorted_idx.clear()
            for i in range(n_exist_trials):
                scaled_obj = self._direction_multiplier * obj_ptr[i]
                sorted_size = <int>self._sorted_obj.size()
                lo = 0
                hi = sorted_size
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if self._sorted_obj[mid] < scaled_obj:
                        lo = mid + 1
                    else:
                        hi = mid
                self._sorted_obj.insert(self._sorted_obj.begin() + lo, scaled_obj)
                self._sorted_idx.insert(self._sorted_idx.begin() + lo, <long>i)
        else:
            self._sorted_obj.clear()
            self._sorted_idx.clear()

        final_noise_val = <double>self.final_noise
        noise_range = self.initial_noise - final_noise_val

        for iteration in range(n_exist_trials, total_trials):
            start_time = perf_counter()
            progress = (<double>iteration + 1.0) / <double>self.n_trials
            self._progress = progress

            for py_var in self._variables.values():
                if py_var.type is list:
                    py_var.values[iteration] = -1
                else:
                    py_var.values[iteration] = np.nan

            if iteration >= self.n_init_points:
                self._current_n_elites = max(
                    1, <int>cround(elite_scale * progress * (1.0 - progress))
                )

                cos_anneal = (1.0 + ccos(M_PI * progress)) * 0.5
                self._current_noise = final_noise_val + noise_range * cos_anneal
                self._current_cat_temp = 1.0 / (0.1 + 0.9 * cos_anneal)

                # Incremental sorted elite selection
                window_start = 0
                if self.elite_window is not None:
                    window_start = max(0, iteration - <int>self.elite_window)

                # Collect top-k from sorted list, respecting window
                sorted_size = <int>self._sorted_obj.size()
                k = self._current_n_elites
                self._elite_buf.clear()
                for i in range(sorted_size):
                    if self._sorted_idx[i] >= window_start:
                        self._elite_buf.push_back(<double>self._sorted_idx[i])
                        if <int>self._elite_buf.size() >= k:
                            break

                n_valid = <int>self._elite_buf.size()
                self._elite_indices_arr = np.empty(n_valid, dtype=np.int64)
                for i in range(n_valid):
                    (<long*>cnp.PyArray_DATA(self._elite_indices_arr))[i] = <long>self._elite_buf[i]

                eps_t = self.epsilon / (<double>iteration + 1.0)
                self._force_random = _rng_double(bg) < eps_t
            else:
                self._force_random = False

            trial = Trial(self, iteration)
            self._current_trial = trial
            obj_value = objective_function(trial)

            if not isinstance(obj_value, (int, float)):
                raise TypeError(
                    "Currently, only numerical outputs (int or float) are supported, but the function "
                    f"returned a value of type {type(obj_value)}. Please ensure that the function returns a "
                    "numerical value."
                )

            if not np.isfinite(obj_value):
                raise ValueError(
                    f"The objective function returned a non-finite value: {obj_value}. "
                    "Please ensure that the function returns a finite numerical value (not NaN or Inf)."
                )

            obj_ptr[iteration] = obj_value
            time_ptr[iteration] = perf_counter() - start_time
            self._trials.append(trial)

            # Insert into sorted elite list (binary search + insert)
            scaled_obj = self._direction_multiplier * obj_value
            sorted_size = <int>self._sorted_obj.size()
            lo = 0
            hi = sorted_size
            while lo < hi:
                mid = (lo + hi) >> 1
                if self._sorted_obj[mid] < scaled_obj:
                    lo = mid + 1
                else:
                    hi = mid
            self._sorted_obj.insert(self._sorted_obj.begin() + lo, scaled_obj)
            self._sorted_idx.insert(self._sorted_idx.begin() + lo, <long>iteration)

            is_new_best = (
                (self.direction == "minimize" and obj_value < best_value) or
                (self.direction == "maximize" and obj_value > best_value)
            )
            if is_new_best:
                old_best_idx = self._best_ever_idx
                best_value = obj_value
                best_iteration = iteration
                self._best_ever_idx = iteration

                if old_best_idx is not None:
                    for var_name in self._variables:
                        py_v = self._variables[var_name]
                        if py_v.type is not float:
                            continue
                        old_val = py_v.values[old_best_idx]
                        new_val = py_v.values[iteration]
                        if old_val != old_val or new_val != new_val:
                            continue
                        prev = self._evo_path.get(var_name, 0.0)
                        self._evo_path[var_name] = 0.8 * prev + 0.2 * (new_val - old_val)

            if self.verbose:
                self._logger.log_trial(
                    iteration=iteration + 1,
                    variables=trial.variables,
                    objective=obj_value,
                    best_value=best_value,
                    best_iteration=best_iteration + 1,
                )

        return

    @staticmethod
    def _validate_init_params(
        n_init_points=None, random_state=None, initial_noise=0.33,
        final_noise=None, direction="minimize", verbose=True,
        epsilon=1.0, elite_window=None,
    ):
        if not isinstance(epsilon, (int, float)):
            raise TypeError(f"epsilon must be a number, got {type(epsilon)}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if elite_window is not None:
            if not isinstance(elite_window, int):
                raise TypeError(f"elite_window must be an integer, got {type(elite_window)}")
            if elite_window <= 0:
                raise ValueError(f"elite_window must be positive, got {elite_window}")
        if n_init_points is not None:
            if not isinstance(n_init_points, int):
                raise TypeError(f"n_init_points must be an integer, got {type(n_init_points)}")
            if n_init_points <= 0:
                raise ValueError(f"n_init_points must be positive, got {n_init_points}")
        if random_state is not None:
            if not isinstance(random_state, int):
                raise TypeError(f"random_state must be None or an integer, got {type(random_state)}")
            if random_state < 0 or random_state > np.iinfo(np.uint32).max:
                raise ValueError(
                    f"random_state must be an None or integer between 0 and {np.iinfo(np.uint32).max}, inclusive. Got {random_state} instead."
                )
        if not isinstance(initial_noise, (int, float)):
            raise TypeError(f"initial_noise must be a number, got {type(initial_noise)}")
        if not (0 < initial_noise <= 1):
            raise ValueError(f"initial_noise must be between 0 and 1 (exclusive), got {initial_noise}")
        if final_noise is not None:
            if not isinstance(final_noise, (int, float)):
                raise TypeError(f"final_noise must be a number, got {type(final_noise)}")
            if not (0 < final_noise <= 1):
                raise ValueError(f"final_noise must be between 0 and 1 (exclusive), got {final_noise}")
            if final_noise > initial_noise:
                raise ValueError(
                    f"final_noise ({final_noise}) must be less than or equal to initial_noise ({initial_noise})"
                )
        if not isinstance(direction, str):
            raise TypeError(f"direction must be a string, got {type(direction)}")
        if direction not in ["minimize", "maximize"]:
            raise ValueError(f"direction must be either 'minimize' or 'maximize', got {direction}")
        if not isinstance(verbose, bool):
            raise TypeError(f"verbose must be a boolean, got {type(verbose)}")

    @property
    def best_trial(self):
        if not self._trials:
            raise ValueError("At least one iteration must be completed before accessing best trial.")
        cdef int n_completed = len(self._trials)
        obj_vals = self._objective_values_arr[:n_completed]
        if self.direction == "minimize":
            best_iter = int(np.argmin(obj_vals))
        else:
            best_iter = int(np.argmax(obj_vals))
        best_trial_obj = self._trials[best_iter]
        variables = {}
        for var_name in self._variables:
            py_v = self._variables[var_name]
            if py_v.type in (int, float):
                val = py_v.values[best_iter]
                if val == val:
                    variables[var_name] = int(val) if py_v.type is int else float(val)
            else:
                cat_idx = int(py_v.values[best_iter])
                if cat_idx >= 0:
                    variables[var_name] = py_v.category_indexer.get_strings(cat_idx)
        return {
            "iteration": best_iter + 1,
            "objective_value": float(self._objective_values_arr[best_iter]),
            "trial_time": float(self._elapsed_times_arr[best_iter]),
            "variables": variables,
            "user_attrs": best_trial_obj.user_attrs,
        }

    @property
    def trials(self):
        if not self._trials:
            raise ValueError("At least one iteration must be completed before accessing trials.")
        cdef int n_completed = len(self._trials)
        history = []
        for iteration in range(n_completed):
            trial = self._trials[iteration]
            trial_dict = {
                "iteration": iteration + 1,
                "objective_value": float(self._objective_values_arr[iteration]),
                "trial_time": float(self._elapsed_times_arr[iteration]),
                "variables": {},
                "user_attrs": trial.user_attrs,
            }
            for var_name in self._variables:
                py_v = self._variables[var_name]
                var_value = py_v.values[iteration]
                if py_v.type is int:
                    if var_value == var_value:
                        trial_dict["variables"][var_name] = int(var_value)
                elif py_v.type is float:
                    if var_value == var_value:
                        trial_dict["variables"][var_name] = float(var_value)
                else:
                    cat_idx = int(var_value)
                    if cat_idx >= 0:
                        trial_dict["variables"][var_name] = py_v.category_indexer.get_strings(cat_idx)
            history.append(trial_dict)
        return history

    @property
    def objective_values(self):
        if not self._trials:
            raise ValueError("At least one iteration must be completed before accessing objective values.")
        return self._objective_values_arr[:len(self._trials)]

    @property
    def elapsed_times(self):
        if not self._trials:
            raise ValueError("At least one iteration must be completed before accessing elapsed times.")
        return self._elapsed_times_arr[:len(self._trials)]
