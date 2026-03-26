from typing import Dict, List, Any, Optional, Callable, Tuple, Union
import math
import numpy as np
from numpy.typing import NDArray
from time import perf_counter

from .variable import Variable
from .logger import OptimizationLogger

from functools import lru_cache


class Trial:
    __slots__ = [
        "study",
        "trial_id",
        "variables",
        "_validated_variables",
        "user_attrs",
    ]

    """
    Represents a single trial in the optimization process.
    """

    def __init__(self, study: "Study", trial_id: int) -> None:
        """
        Initialize a Trial instance.

        Parameters
        ----------
        study : Study
            The study associated with this trial.
        trial_id : int
            The unique identifier for this trial.
        """
        self.study = study
        self.trial_id = trial_id
        self.variables: Dict[str, Variable] = {}
        self._validated_variables = set()
        self.user_attrs: Dict[str, Any] = {}

    def add_attr(self, name: str, value: Any) -> None:
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

    def __repr__(self) -> str:
        """
        Return a string representation of the Trial instance.

        Returns
        -------
        str
            A string representation of the Trial instance.
        """
        return f"Trial(trial_id={self.trial_id}, variables={self.variables}, user_attrs={self.user_attrs})"

    @staticmethod
    @lru_cache(maxsize=None)
    def _validate_numerical_cached(
        name: str, low: Any, high: Any, expected_type: type, log: bool
    ) -> None:
        """
        Validate numerical variables using cached results.

        Parameters
        ----------
        name : str
            The name of the variable.
        low : Any
            The lower bound of the variable range.
        high : Any
            The upper bound of the variable range.
        expected_type : type
            The expected type of the variable (int or float).
        log : bool
            Whether the variable is log-scaled.
        """
        # Float için hem int hem float kabul edilir
        if expected_type is float:
            if not (isinstance(low, (int, float)) and isinstance(high, (int, float))):
                raise TypeError(
                    f"Variable '{name}': 'low' and 'high' must be numeric, got {type(low)} and {type(high)}"
                )
        # Int için sadece int kabul edilir
        elif expected_type is int:
            if not (isinstance(low, int) and isinstance(high, int)):
                raise TypeError(
                    f"Variable '{name}': 'low' and 'high' must be integers, got {type(low)} and {type(high)}"
                )
        else:
            raise TypeError(f"Variable '{name}': Unsupported type {expected_type}")

        # low ve high değerlerini aynı tipe çevir (float durumu için)
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

    def _validate_numerical(
        self, name: str, low: Any, high: Any, expected_type: type, log: bool
    ) -> None:
        """
        Validate numerical variables and cache the results.

        Parameters
        ----------
        name : str
            The name of the variable.
        low : Any
            The lower bound of the variable range.
        high : Any
            The upper bound of the variable range.
        expected_type : type
            The expected type of the variable (int or float).
        log : bool
            Whether the variable is log-scaled.
        """
        if not isinstance(name, str):
            raise TypeError(f"Variable name must be a string, got {type(name)}")

        if name == "":
            raise ValueError("Variable name cannot be an empty string.")

        self._validate_numerical_cached(name, low, high, expected_type, log)
        self._validated_variables.add(name)

    @staticmethod
    @lru_cache(maxsize=None)
    def _validate_categorical_cached(
        name: str, categories_tuple: Tuple[Any, ...]
    ) -> None:
        """
        Validate categorical variables using cached results.

        Parameters
        ----------
        name : str
            The name of the variable.
        categories_tuple : Tuple[Any, ...]
            A tuple of valid categorical values.
        """
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

    def _validate_categorical(self, name: str, categories: List[Any]) -> None:
        """
        Validate categorical variables and cache the results.

        Parameters
        ----------
        name : str
            The name of the variable.
        categories : List[Any]
            A list of valid categorical values.
        """
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
        self._validate_categorical_cached(name, categories_tuple)
        self._validated_variables.add(name)

    def suggest_float(
        self, name: str, low: float, high: float, log: bool = False
    ) -> float:
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

    def suggest_int(self, name: str, low: int, high: int, log: bool = False) -> int:
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
        value = self.study._suggest_numerical(name, low, high, int, log)
        self.variables[name] = value
        return value

    def suggest_categorical(self, name: str, categories: List[str]) -> str:
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


class Study:
    """
    Mixed Adaptive Random Search for Optimization
    """

    def __init__(
        self,
        initial_noise: float = 0.2,
        direction: str = "minimize",
        n_init_points: Optional[int] = None,
        final_noise: Optional[float] = None,
        exploration_mode: str = "none",
        epsilon: float = 1.0,
        elite_weighting: str = "random",
        rho: float = 0.5,
        elite_window: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the Study.

        Parameters
        ----------
        initial_noise : float, default = 0.2
            Initial noise level.
        direction : str, default = "minimize"
            Direction of optimization, either "minimize" or "maximize".
        n_init_points : int, default = None
            Number of initial random points. If `None`, it is set as:
            round(sqrt(`n_trials`))
        final_noise : float, default = None
            Final noise level. If `None`, it is set as:
            min(2.0 / `n_trials`, `initial_noise`)
        exploration_mode : str, default = "none"
            Global exploration strategy. ``"none"`` disables global exploration.
            ``"epsilon"`` enables epsilon-greedy exploration: at each adaptive
            trial, with probability ``epsilon / (t + 1)`` a uniform random
            sample is drawn instead of the elite-guided step.
        epsilon : float, default = 1.0
            Exploration constant used when ``exploration_mode="epsilon"``.
            Controls the rate eps_t = epsilon / (t + 1). Ignored when
            ``exploration_mode="none"``.
        elite_weighting : str, default = "random"
            How a base value is selected from elites. ``"random"`` picks one
            elite uniformly at random. ``"log_rank"`` computes a weighted mean
            using CMA-ES-style log-rank weights. ``"mixed"`` uses a
            floored-parabolic schedule: rho_t = 0.15 + (rho - 0.15) * 4p(1-p),
            peaking mid-search when the elite pool is largest. With probability
            rho_t uses log-rank weighted mean, otherwise picks a single
            random elite.
        rho : float, default = 0.5
            Peak weighted-mean probability for ``"mixed"`` mode.
            Effective probability follows a parabolic schedule synchronized
            with the elite count.
        elite_window : int, default = None
            If set, only the most recent ``elite_window`` completed trials are
            considered for elite selection. If ``None``, full history is used.
        random_state : int, default = None
            Seed for reproducibility.
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
            exploration_mode=exploration_mode,
            epsilon=epsilon,
            elite_weighting=elite_weighting,
            rho=rho,
            elite_window=elite_window,
        )

        self.n_init_points = n_init_points
        self.initial_noise = initial_noise
        self.verbose = verbose
        self.direction = direction
        self.final_noise = final_noise
        self.exploration_mode = exploration_mode
        self.epsilon = epsilon
        self.elite_weighting = elite_weighting
        self.rho = rho
        self.elite_window = elite_window

        self._rng = np.random.RandomState(random_state)
        self._objective_values: NDArray[np.float64] = None
        self._elapsed_times: NDArray[np.float64] = None
        self._current_trial: Optional[Trial] = None
        self._trials: List[Trial] = []
        self._variables: Dict[str, Variable] = {}

        self._progress: float = None
        self._current_noise: float = None
        self._current_n_elites: float = None
        self._current_cat_temp: float = None
        self._direction_multiplier: float = None
        self._elite_indices: NDArray[np.int64] = None
        self._elite_weights: NDArray[np.float64] = None
        self._force_random: bool = False

        self._use_weighted: bool = False

        self._logger = OptimizationLogger() if verbose else None

    def __repr__(self) -> str:
        return (
            f"Study(n_init_points={self.n_init_points}, "
            f"initial_noise={self.initial_noise}, "
            f"final_noise={self.final_noise}, "
            f"direction='{self.direction}', "
            f"verbose={self.verbose})"
        )

    def _suggest_numerical(
        self,
        name: str,
        low: Union[int, float],
        high: Union[int, float],
        var_type: type,
        log: bool,
    ) -> Union[float, int]:
        """
        Suggests a numerical variable value.

        Parameters
        ----------
        name : str
            The name of the variable.
        low : Union[int, float]
            The lower bound for the variable.
        high : Union[int, float]
            The upper bound for the variable.
        var_type : type
            The type of variable (int or float).
        log : bool
            Whether to sample in logarithmic scale.

        Returns
        -------
        Union[float, int]
            The suggested numerical value.
        """
        var = self._variables.get(name)

        if var is None:
            var = Variable(
                name=name,
            )
            var.set_values(
                max_iter=self.n_trials, var_type_or_categories=var_type
            )
            self._variables[name] = var

        else:
            if var.type != var_type:
                raise TypeError(
                    f"Variable '{name}' has already been registered with type {var.type}, "
                    f"but an attempt was made to register it as type {var_type}. Ensure consistency."
                )

        trial_id = self._current_trial.trial_id

        if trial_id < self.n_init_points or self._force_random:
            value = self._sample_value(low, high, log)

        else:
            var_values = var.values[:trial_id]
            elite_var = var_values[self._elite_indices]
            in_bounds = (elite_var >= low) & (elite_var <= high)

            if np.any(in_bounds):
                if self._use_weighted and self._elite_weights is not None:
                    w = self._elite_weights[in_bounds]
                    w = w / w.sum()
                    base_value = np.dot(w, elite_var[in_bounds])
                else:
                    base_value = self._rng.choice(elite_var[in_bounds])
            else:
                range_mask = (var_values >= low) & (var_values <= high)
                n_in_bounds = int(range_mask.sum())
                if n_in_bounds == 0:
                    value = self._sample_value(low, high, log)
                    if var_type is int:
                        value = int(value) + int(
                            (self._rng.random() < abs(value - int(value)))
                            * (1 if value > 0 else -1)
                        )
                    var.values[trial_id] = value
                    return value
                masked_var = var_values[range_mask]
                n_elites = min(self._current_n_elites, n_in_bounds)
                if n_elites >= n_in_bounds:
                    base_value = self._rng.choice(masked_var)
                else:
                    masked_obj = (
                        self._direction_multiplier
                        * self._objective_values[:trial_id][range_mask]
                    )
                    top_k = np.argpartition(masked_obj, n_elites - 1)[:n_elites]
                    base_value = self._rng.choice(masked_var[top_k])

            if log:
                log_base = np.log(base_value)
                log_high = np.log(high)
                log_low = np.log(low)
                log_range = log_high - log_low
                noise = self._rng.normal(
                    loc=0.0, scale=self._current_noise * log_range
                )

                value = np.exp(
                    self._reflect_at_boundaries(log_base + noise, log_low, log_high)
                )

            else:
                var_range = high - low
                noise = self._rng.normal(
                    loc=0.0, scale=self._current_noise * var_range
                )

                value = self._reflect_at_boundaries(base_value + noise, low, high)

        if var_type is int:
            value = int(value) + int((self._rng.random() < abs(value - int(value))) * (1 if value > 0 else -1))

        var.values[self._current_trial.trial_id] = value

        return value

    def _suggest_categorical(self, name: str, categories: List[str]) -> Any:
        """
        Suggests a categorical variable value.

        Parameters
        ----------
        name : str
            The name of the variable.
        categories : List[Any]
            A list of possible categorical values.

        Returns
        -------
        Any
            The suggested categorical value.
        """
        var = self._variables.get(name)
        trial_id = self._current_trial.trial_id

        if var is None:
            var = Variable(name=name)
            var.set_values(
                max_iter=self.n_trials, var_type_or_categories=categories
            )
            self._variables[name] = var
        else:
            if var.type is not type(categories):
                raise TypeError(
                    f"Variable '{name}' has already been registered with type {var.type}, "
                    f"but an attempt was made to register it as type {type(categories)}. Ensure consistency."
                )

        cat_indices = var.category_indexer.get_indices(categories)
        cat_size = cat_indices.size

        # Expand one-hot array if new categories appeared
        if cat_indices.max() + 1 > var.values.shape[1]:
            var.set_values(
                max_iter=self.n_trials, var_type_or_categories=categories
            )

        if trial_id < self.n_init_points or self._force_random:
            category_idx = self._rng.choice(cat_indices)

        else:
            elite_values = var.values[self._elite_indices][:, cat_indices]
            active_mask = np.any(elite_values, axis=1)
            active_elite_values = elite_values[active_mask]

            if active_elite_values.size == 0:
                previous_values = var.values[:trial_id][:, cat_indices]
                active_previous_values = previous_values[np.any(previous_values, axis=1)]

                if active_previous_values.size == 0:
                    category_idx = self._rng.choice(cat_indices)
                else:
                    active_elite_values = active_previous_values
                    active_mask = None  # weights not applicable for fallback

            if active_elite_values.size != 0:
                noise = self._rng.normal(
                    loc=0.0, scale=self._current_noise, size=cat_size
                )

                if self._use_weighted and self._elite_weights is not None and active_mask is not None:
                    w = self._elite_weights[active_mask]
                    w = w / w.sum()
                    elite_mean = np.average(active_elite_values, axis=0, weights=w)
                else:
                    idx = self._rng.randint(len(active_elite_values))
                    elite_mean = active_elite_values[idx].astype(np.float64)

                chosen_elites_with_noise = elite_mean + noise

                while True:
                    below = chosen_elites_with_noise < 0.0
                    above = chosen_elites_with_noise > 1.0
                    if not (np.any(below) or np.any(above)):
                        break
                    chosen_elites_with_noise = np.where(
                        below, -chosen_elites_with_noise / 2.0, chosen_elites_with_noise
                    )
                    chosen_elites_with_noise = np.where(
                        above,
                        1.0 - (chosen_elites_with_noise - 1.0) / 2.0,
                        chosen_elites_with_noise,
                    )

                exps = np.exp(
                    (chosen_elites_with_noise - chosen_elites_with_noise.max())
                    * self._current_cat_temp
                )
                probs = exps / exps.sum()

                category_idx = cat_indices[self._rng.choice(cat_size, p=probs)]

        result = np.zeros(len(var.category_indexer), dtype=np.float64)
        result[category_idx] = 1.0

        var.values[trial_id, :] = result

        return var.category_indexer.get_strings(category_idx)

    @staticmethod
    def _reflect_at_boundaries(x: float, low: float = 0.0, high: float = 1.0) -> float:
        """
        Reflects values exceeding boundaries back into a valid range.

        Parameters
        ----------
        x : float
            Input value.
        low : float, optional, default = 0.0
            Lower boundary.
        high : float, optional, default = 1.0
            Upper boundary.

        Returns
        -------
        float
            Value reflected into the valid range.
        """
        while True:
            if x < low:
                x = low + (low - x) / 2.0
            elif x > high:
                x = high - (x - high) / 2.0
            else:
                break
        return x

    def _sample_value(self, low: float, high: float, log: bool) -> float:
        """
        Samples a numerical value within the specified range.

        This function generates a random value between `low` and `high`. If `log` is True,
        the sampling is done in logarithmic space, ensuring a proper distribution
        when dealing with exponentially scaled variables.

        Parameters
        ----------
        low : float
            The lower bound of the sampling range.
        high : float
            The upper bound of the sampling range.
        log : bool
            Whether to sample in logarithmic scale.

        Returns
        -------
        float
            A randomly sampled value within the specified range.
        """

        if log:
            return np.exp(self._rng.uniform(np.log(low), np.log(high)))
        else:
            return self._rng.uniform(low, high)

    def optimize(
        self, objective_function: Callable[[Trial], Union[float, int]], n_trials: int
    ) -> None:
        """
        Runs the optimization loop.

        Parameters
        ----------
        objective_function : Callable[[Trial], Union[float, int]]
            The function to optimize.
        n_trials : int
            The number of trials.

        Returns
        -------
        None
        """
        if not isinstance(n_trials, int):
            raise TypeError("n_trials must be an integer.")

        if n_trials <= 0:
            raise ValueError("n_trials must be a positive integer.")

        if not callable(objective_function):
            raise TypeError("objective_function must be a callable function.")

        if self._trials:
            n_exist_trials = len(self._trials)

            # Find best iteration based on direction
            if self.direction == "minimize":
                best_iteration = np.argmin(self._objective_values[:n_exist_trials])
                best_value = self._objective_values[best_iteration]
            else:
                best_iteration = np.argmax(self._objective_values[:n_exist_trials])
                best_value = self._objective_values[best_iteration]

            total_trials = n_trials + n_exist_trials
            self.n_trials = total_trials

            old_objective_values = self._objective_values
            old_elapsed_times = self._elapsed_times

            self._objective_values = np.empty(shape=(total_trials,), dtype=np.float64)
            self._elapsed_times = np.empty(shape=(total_trials,), dtype=np.float64)

            # Copy existing data
            self._objective_values[:n_exist_trials] = old_objective_values[
                :n_exist_trials
            ]
            self._elapsed_times[:n_exist_trials] = old_elapsed_times[:n_exist_trials]

            for var in self._variables.keys():
                self._variables[var].add_iter(n_trials)

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
                self.final_noise = min(2.0 / n_trials, self.initial_noise)

            self.n_trials = n_trials
            self._objective_values = np.empty(shape=(n_trials,), dtype=np.float64)
            self._elapsed_times = np.empty(shape=(n_trials,), dtype=np.float64)

            if self.n_init_points is None:
                self.n_init_points = max(10, round(np.sqrt(self.n_trials)))

        elite_scale: float = 2.0 * math.sqrt(total_trials)
        self._direction_multiplier = 1.0 if self.direction == "minimize" else -1.0

        noise_range = self.initial_noise - self.final_noise

        # Start from the existing trials count
        for iteration in range(n_exist_trials, total_trials):
            start_time = perf_counter()
            self._progress = (iteration + 1) / self.n_trials

            for var in self._variables.values():
                if var.values.ndim == 1:
                    var.values[iteration] = np.nan
                else:
                    var.values[iteration, :] = 0.0

            if iteration >= self.n_init_points:
                self._current_n_elites = max(
                    1, round(elite_scale * self._progress * (1 - self._progress))
                )

                cos_anneal = (1.0 + math.cos(math.pi * self._progress)) * 0.5

                self._current_noise = self.final_noise + noise_range * cos_anneal

                self._current_cat_temp = 1.0 / (0.1 + 0.9 * cos_anneal)

                # Elite selection with optional window
                window_start = 0
                if self.elite_window is not None:
                    window_start = max(0, iteration - self.elite_window)

                obj_scaled = (
                    self._direction_multiplier
                    * self._objective_values[window_start:iteration]
                )
                pool_size = len(obj_scaled)
                k = min(self._current_n_elites, pool_size)

                if k >= pool_size:
                    self._elite_indices = np.arange(window_start, iteration)
                else:
                    local_idx = np.argpartition(obj_scaled, k - 1)[:k]
                    self._elite_indices = local_idx + window_start

                # Compute elite weights
                if self.elite_weighting in ("log_rank", "mixed") and k > 1:
                    elite_obj = (
                        self._direction_multiplier
                        * self._objective_values[self._elite_indices]
                    )
                    rank_order = np.argsort(elite_obj)
                    w = np.log(k + 0.5) - np.log(np.arange(1, k + 1))
                    w /= w.sum()
                    self._elite_weights = w
                    self._elite_indices = self._elite_indices[rank_order]
                else:
                    self._elite_weights = None

                # Epsilon-exploration
                if self.exploration_mode == "epsilon":
                    eps_t = self.epsilon / (iteration + 1)
                    self._force_random = self._rng.random() < eps_t
                else:
                    self._force_random = False

                # Per-trial weighted decision (one flip per trial)
                # Floored parabolic: peaks mid-search, floor at 0.15
                if self.elite_weighting == "mixed":
                    parabola = 4.0 * self._progress * (1.0 - self._progress)
                    rho_t = 0.15 + (self.rho - 0.15) * parabola
                    self._use_weighted = self._rng.random() < rho_t
                elif self.elite_weighting == "log_rank":
                    self._use_weighted = True
                else:
                    self._use_weighted = False
            else:
                self._force_random = False
                self._use_weighted = False

            trial = Trial(self, iteration)
            self._current_trial = trial
            obj_value: float = objective_function(trial)

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

            self._elapsed_times[iteration] = perf_counter() - start_time
            self._objective_values[iteration] = obj_value
            self._trials.append(trial)

            improved = (self.direction == "minimize" and obj_value < best_value) or (
                self.direction == "maximize" and obj_value > best_value
            )

            # Update best value based on optimization direction
            if improved:
                best_value = obj_value
                best_iteration = iteration

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
        n_init_points: Any,
        random_state: Any,
        initial_noise: Any,
        final_noise: Any,
        direction: Any,
        verbose: Any,
        exploration_mode: Any = "none",
        epsilon: Any = 1.0,
        elite_weighting: Any = "random",
        rho: Any = 0.5,
        elite_window: Any = None,
    ) -> None:
        """
        Validates initialization variables for Study.
        """
        # exploration_mode validation
        if exploration_mode not in ("none", "epsilon"):
            raise ValueError(
                f"exploration_mode must be 'none' or 'epsilon', got '{exploration_mode}'"
            )

        # epsilon validation
        if not isinstance(epsilon, (int, float)):
            raise TypeError(f"epsilon must be a number, got {type(epsilon)}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # elite_weighting validation
        if elite_weighting not in ("random", "log_rank", "mixed"):
            raise ValueError(
                f"elite_weighting must be 'random', 'log_rank', or 'mixed', got '{elite_weighting}'"
            )

        # rho validation
        if not isinstance(rho, (int, float)):
            raise TypeError(f"rho must be a number, got {type(rho)}")
        if not (0.15 <= rho <= 1):
            raise ValueError(f"rho must be between 0.15 and 1, got {rho}")

        # elite_window validation
        if elite_window is not None:
            if not isinstance(elite_window, int):
                raise TypeError(
                    f"elite_window must be an integer, got {type(elite_window)}"
                )
            if elite_window <= 0:
                raise ValueError(
                    f"elite_window must be positive, got {elite_window}"
                )
        # n_init_points validation
        if n_init_points is not None:
            if not isinstance(n_init_points, int):
                raise TypeError(
                    f"n_init_points must be an integer, got {type(n_init_points)}"
                )
            if n_init_points <= 0:
                raise ValueError(f"n_init_points must be positive, got {n_init_points}")

        # random_state validation
        if random_state is not None:
            if not isinstance(random_state, int):
                raise TypeError(
                    f"random_state must be None or an integer, got {type(random_state)}"
                )

            if random_state < 0 or random_state > np.iinfo(np.uint32).max:
                raise ValueError(
                    f"random_state must be an None or integer between 0 and {np.iinfo(np.uint32).max}, inclusive. Got {random_state} instead."
                )

        # initial_noise validation
        if not isinstance(initial_noise, (int, float)):
            raise TypeError(
                f"initial_noise must be a number, got {type(initial_noise)}"
            )
        if not (0 < initial_noise <= 1):
            raise ValueError(
                f"initial_noise must be between 0 and 1 (exclusive), got {initial_noise}"
            )

        # final_noise validation
        if final_noise is not None:
            if not isinstance(final_noise, (int, float)):
                raise TypeError(
                    f"final_noise must be a number, got {type(final_noise)}"
                )
            if not (0 < final_noise <= 1):
                raise ValueError(
                    f"final_noise must be between 0 and 1 (exclusive), got {final_noise}"
                )
            if final_noise > initial_noise:
                raise ValueError(
                    f"final_noise ({final_noise}) must be less than or equal to initial_noise ({initial_noise})"
                )

        # direction validation
        if not isinstance(direction, str):
            raise TypeError(f"direction must be a string, got {type(direction)}")
        if direction not in ["minimize", "maximize"]:
            raise ValueError(
                f"direction must be either 'minimize' or 'maximize', got {direction}"
            )

        # verbose validation
        if not isinstance(verbose, bool):
            raise TypeError(f"verbose must be a boolean, got {type(verbose)}")

    @property
    def best_trial(self) -> Dict[str, Any]:
        """
        Get the best trial's details, including the iteration number,
        objective value, execution time, and variable values.

        The best trial is determined based on the optimization direction
        ('minimize' or 'maximize').

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the following keys:

            - **iteration** (:obj:`int`)

            The iteration number of the best trial.

            - **objective_value** (:obj:`float`)

            The best recorded objective function value.

            - **trial_time** (:obj:`float`)

            The execution time of the best trial in seconds.

            - **variable** (:obj:`Dict[str, Union[int, float, str]]`)

            A dictionary of variable values from the best trial. Keys are variable names,
            and values are their respective values (int, float, or categorical as a string).

            - **user_attrs** (:obj:`Dict[str, Any]`)

            A dictionary of user-defined attributes for the best trial.

        """
        if not self._trials:
            raise ValueError(
                "At least one iteration must be completed before accessing best trial."
            )

        n_completed = len(self._trials)
        obj_vals = self._objective_values[:n_completed]

        if self.direction == "minimize":
            best_iteration = int(np.argmin(obj_vals))
        else:
            best_iteration = int(np.argmax(obj_vals))

        best_trial = self._trials[best_iteration]

        variables = {}
        for var_name, var in self._variables.items():
            if var.type in (int, float):
                if not np.isnan(var.values[best_iteration]):
                    if var.type is int:
                        variables[var_name] = int(var.values[best_iteration])
                    else:
                        variables[var_name] = float(var.values[best_iteration])
            else:
                if np.any(var.values[best_iteration]):
                    variables[var_name] = var.category_indexer.get_strings(
                        np.argmax(var.values[best_iteration])
                    )

        return {
            "iteration": best_iteration + 1,
            "objective_value": float(self._objective_values[best_iteration]),
            "trial_time": float(self._elapsed_times[best_iteration]),
            "variables": variables,
            "user_attrs": best_trial.user_attrs,
        }

    @property
    def trials(self) -> List[Dict[str, Any]]:
        """
        Get the complete history of all trials in the optimization process.

        Each trial includes its iteration number, objective function value, execution time,
        and variable values.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries, where each dictionary represents a trial with the following keys:

            - **iteration** (:obj:`int`)

            The iteration number of the trial.

            - **objective_value** (:obj:`float`)

            The objective function value for the trial.

            - **trial_time** (:obj:`float`)

            The execution time of the trial in seconds.

            - **variables** (:obj:`Dict[str, Union[int, float, str]]`)

            A dictionary of variable values from the trial. Keys are variable names,
            and values are their respective values (int, float, or categorical as a string).

            - **user_attrs** (:obj:`Dict[str, Any]`)

            A dictionary of user-defined attributes for the trial.
        """
        if not self._trials:
            raise ValueError(
                "At least one iteration must be completed before accessing trials."
            )

        n_completed = len(self._trials)
        history = []

        for iteration in range(n_completed):
            trial = self._trials[iteration]
            trial_dict = {
                "iteration": iteration + 1,
                "objective_value": float(self._objective_values[iteration]),
                "trial_time": float(self._elapsed_times[iteration]),
                "variables": {},
                "user_attrs": trial.user_attrs,
            }

            # Store variable values for this iteration
            for var_name, var in self._variables.items():
                # Skip variables that weren't used in this trial
                var_value = var.values[iteration]

                if var.type is int:
                    if not np.isnan(var_value):
                        trial_dict["variables"][var_name] = int(var_value)
                elif var.type is float:
                    if not np.isnan(var_value):
                        trial_dict["variables"][var_name] = float(var_value)
                else:  # categorical variables
                    # For categorical variables, check if any value is non-zero
                    if np.any(var_value):
                        trial_dict["variables"][var_name] = (
                            var.category_indexer.get_strings(np.argmax(var_value))
                        )

            history.append(trial_dict)

        return history

    @property
    def objective_values(self) -> NDArray[np.float64]:
        """
        Returns the objective function values for all completed trials.

        This property provides an array where each element represents the
        objective function value obtained at a specific trial.

        Returns
        -------
        NDArray[np.float64]
            A NumPy array containing the recorded objective function values for
            all trials, ordered by their trial index.
        """
        if not self._trials:
            raise ValueError(
                "At least one iteration must be completed before accessing objective values."
            )
        else:
            return self._objective_values[: len(self._trials)]

    @property
    def elapsed_times(self) -> NDArray[np.float64]:
        """
        Returns the execution times of all completed trials.

        This property provides an array where each element represents the time taken
        to evaluate the objective function for a specific trial.

        Returns
        -------
        NDArray[np.float64]
            A NumPy array containing the recorded execution times (in seconds)
            for each trial, ordered by their trial index.
        """
        if not self._trials:
            raise ValueError(
                "At least one iteration must be completed before accessing elapsed times."
            )
        else:
            return self._elapsed_times[: len(self._trials)]
