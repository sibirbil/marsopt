from typing import Dict, List, Any, Optional, Callable, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from time import perf_counter

from .parameters import Parameter
from .logger import OptimizationLogger

from functools import lru_cache


class Trial:
    __slots__ = ["optimizer", "trial_id", "params", "_validated_params"]

    """
    Represents a single trial in the optimization process.
    """

    def __init__(self, optimizer: "MARSOpt", trial_id: int) -> None:
        self.optimizer = optimizer
        self.trial_id = trial_id
        self.params: Dict[str, Any] = {}
        self._validated_params = set()

    def __repr__(self) -> str:
        return f"Trial(trial_id={self.trial_id}, params={self.params})"

    @staticmethod
    @lru_cache(maxsize=None)
    def _validate_numerical_cached(
        name: str, low: Any, high: Any, expected_type: type, log: bool
    ) -> None:
        """Cached validation for numerical parameters."""
        # Float için hem int hem float kabul edilir
        if expected_type is float:
            if not (isinstance(low, (int, float)) and isinstance(high, (int, float))):
                raise TypeError(
                    f"Parameter '{name}': 'low' and 'high' must be numeric, got {type(low)} and {type(high)}"
                )
        # Int için sadece int kabul edilir
        elif expected_type is int:
            if not (isinstance(low, int) and isinstance(high, int)):
                raise TypeError(
                    f"Parameter '{name}': 'low' and 'high' must be integers, got {type(low)} and {type(high)}"
                )
        else:
            raise TypeError(f"Parameter '{name}': Unsupported type {expected_type}")

        # low ve high değerlerini aynı tipe çevir (float durumu için)
        low = expected_type(low)
        high = expected_type(high)

        if low >= high:
            raise ValueError(
                f"Parameter '{name}': 'low' must be less than 'high' (got {low} >= {high})"
            )
        if log and (low <= 0 or high <= 0):
            raise ValueError(
                f"Parameter '{name}': 'low' and 'high' must be positive when 'log' is True (got {low}, {high})"
            )

    def _validate_numerical(
        self, name: str, low: Any, high: Any, expected_type: type, log: bool
    ) -> None:
        """Validates numerical parameters using cached results."""
        if not isinstance(name, str):
            raise TypeError(f"Parameter name must be a string, got {type(name)}")

        self._validate_numerical_cached(name, low, high, expected_type, log)
        self._validated_params.add(name)

    @staticmethod
    @lru_cache(maxsize=None)
    def _validate_categorical_cached(
        name: str, categories_tuple: Tuple[Any, ...]
    ) -> None:
        """Cached validation for categorical parameters."""
        if len(categories_tuple) < 2:
            raise ValueError(
                f"Parameter '{name}': 'categories' must contain at least two elements"
            )

        if len(set(categories_tuple)) != len(categories_tuple):
            raise ValueError(
                f"Parameter '{name}': 'categories' contains duplicate values"
            )

        try:
            _ = categories_tuple[0]
        except (TypeError, IndexError):
            raise TypeError(
                f"Parameter '{name}': 'categories' must be indexable, got {type(categories_tuple)} with non-indexable elements"
            )

    def _validate_categorical(self, name: str, categories: List[Any]) -> None:
        """Validates categorical parameters using cached results."""
        if not isinstance(name, str):
            raise TypeError(f"Parameter name must be a string, got {type(name)}")

        if not isinstance(categories, list):
            raise TypeError(
                f"Parameter '{name}': 'categories' must be a list, got {type(categories)}"
            )

        categories_tuple = tuple(categories)
        self._validate_categorical_cached(name, categories_tuple)
        self._validated_params.add(name)

    def suggest_float(
        self, name: str, low: float, high: float, log: bool = False
    ) -> float:
        """
        Suggests a floating-point parameter value.
        Accepts both int and float inputs for low and high.
        """
        self._validate_numerical(name, low, high, float, log)
        value = self.optimizer._suggest_numerical(name, low, high, float, log)
        self.params[name] = value
        return value

    def suggest_int(self, name: str, low: int, high: int, log: bool = False) -> int:
        """
        Suggests an integer parameter value.
        Only accepts integer inputs for low and high.
        """
        self._validate_numerical(name, low, high, int, log)
        value = self.optimizer._suggest_numerical(name, low, high, int, log)
        self.params[name] = value
        return value

    def suggest_categorical(self, name: str, categories: List[Any]) -> Any:
        """Suggests a categorical parameter value."""
        self._validate_categorical(name, categories)
        value = self.optimizer._suggest_categorical(name, categories)
        self.params[name] = value
        return value


class MARSOpt:
    """
    A global optimization algorithm that searches the parameter space.

    Attributes
    ----------
    n_init_points : int
        Number of initial random trials before optimization.
    initial_noise : float
        Initial noise level in parameter selection.
    rng : np.random.Generator
        Random number generator for reproducibility.
    parameters : Dict[str, Parameter]
        Dictionary storing parameter definitions.
    objective_values : NDArray
        Array storing objective function values across trials.
    current_trial : Optional[Trial]
        The current trial being evaluated.
    """

    def __init__(
        self,
        initial_noise: float = 0.2,
        direction: str = "minimize",
        n_init_points: Optional[int] = None,
        final_noise: Optional[float] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initializes the optimizer.

        Parameters
        ----------
        n_init_points : int, optional
            Number of initial random points (default is 10).
        random_state : Optional[int], optional
            Seed for reproducibility (default is None).
        initial_noise : float, optional
            Initial noise level (default is 0.20).
        """

        self._validate_init_params(
            n_init_points=n_init_points,
            random_state=random_state,
            final_noise=final_noise,
            initial_noise=initial_noise,
            direction=direction,
            verbose=verbose,
        )

        self.n_init_points = n_init_points
        self.initial_noise = initial_noise
        self.rng = np.random.default_rng(random_state)
        self.direction = direction

        self.parameters: Dict[str, Parameter] = {}
        self.objective_values: NDArray = None
        self.trial_times: NDArray = None
        self.current_trial: Optional[Trial] = None
        self.verbose = verbose
        self.final_noise = final_noise

        self._progress: float = None
        self._current_noise: float = None
        self._current_n_elites: float = None
        self._current_cat_temp: float = None
        self._obj_arg_sort: NDArray = None
        self._logger = OptimizationLogger(name="MARSOpt") if verbose else None

        self._best_value: int = None

    def __repr__(self) -> str:
        return (
            f"MARSOpt(n_init_points={self.n_init_points}, "
            f"initial_noise={self.initial_noise}, "
            f"verbose={self.verbose})"
        )

    def _suggest_numerical(
        self, name: str, low: float, high: float, param_type: type, log: bool
    ) -> float:
        """
        Suggests a numerical parameter value.

        Parameters
        ----------
        name : str
            The name of the parameter.
        low : float
            The lower bound for the parameter.
        high : float
            The upper bound for the parameter.
        param_type : type
            The type of parameter (int or float).
        log : bool
            Whether to sample in logarithmic scale.

        Returns
        -------
        float
            The suggested numerical value.
        """
        param = self.parameters.get(name)

        if param is None:
            param = Parameter(
                name=name,
            )
            param.set_values(
                max_iter=self.n_trials, param_type_or_categories=param_type
            )
            self.parameters[name] = param

        if self.current_trial.trial_id < self.n_init_points:
            value = self._sample_value(low, high, log)

        else:
            param_values = param.values[: self.current_trial.trial_id]
            range_mask = (param_values >= low) & (param_values <= high)

            if not np.any(range_mask):
                value = self._sample_value(low, high, log)

            else:
                sorted_indices = self._obj_arg_sort[range_mask[self._obj_arg_sort]]
                values_masked = param_values[sorted_indices]
                base_value = self.rng.choice(values_masked[: self._current_n_elites])

                if log:
                    log_base = np.log(base_value)
                    log_high = np.log(high)
                    log_low = np.log(low)
                    log_range = log_high - log_low
                    noise = self.rng.normal(
                        loc=0.0, scale=self._current_noise * log_range
                    )

                    value = np.exp(
                        self._reflect_at_boundaries(log_base + noise, log_low, log_high)
                    )

                else:
                    # Apply noise directly
                    param_range = high - low
                    noise = self.rng.normal(
                        loc=0.0, scale=self._current_noise * param_range
                    )

                    value = self._reflect_at_boundaries(base_value + noise, low, high)

            if param_type == int:
                # probabilistic rounding
                frac = value - np.floor(value)
                if self.rng.random() < frac:
                    value = int(np.ceil(value))
                else:
                    value = int(np.floor(value))

        param.values[self.current_trial.trial_id] = value

        return value

    def _suggest_categorical(self, name: str, categories: List[str]) -> Any:
        """
        Suggests a categorical parameter value.

        Parameters
        ----------
        name : str
            The name of the parameter.
        categories : List[Any]
            A list of possible categorical values.

        Returns
        -------
        Any
            The suggested categorical value.
        """
        param = self.parameters.get(name)
        trial_id = self.current_trial.trial_id

        if param is None:
            param = Parameter(name=name)
            param.set_values(
                max_iter=self.n_trials, param_type_or_categories=categories
            )
            self.parameters[name] = param

        else:
            param.set_values(
                max_iter=self.n_trials, param_type_or_categories=categories
            )

        cat_indices = param.category_indexer.get_indices(categories)
        cat_size = cat_indices.size

        if trial_id < self.n_init_points:
            category_idx = self.rng.choice(cat_indices)

        else:
            sorted_trials = self._obj_arg_sort[: self._current_n_elites]

            # Get parameter values for the best trials
            param_values = param.values[sorted_trials[:, np.newaxis], cat_indices]

            noise = self.rng.normal(loc=0.0, scale=self._current_noise, size=cat_size)

            chosen_elites_with_noise = param_values[:, cat_indices].mean(axis=0) + noise

            for i in range(cat_size):
                chosen_elites_with_noise[i] = self._reflect_at_boundaries(
                    chosen_elites_with_noise[i]
                )

            exps = np.exp(
                (chosen_elites_with_noise - np.max(chosen_elites_with_noise))
                * self._current_cat_temp
            )
            probs = exps / exps.sum()

            category_idx = cat_indices[self.rng.choice(cat_size, p=probs)]

        result = np.zeros(len(param.category_indexer), dtype=np.float64)
        result[category_idx] = 1.0

        param.values[trial_id, :] = result

        return param.category_indexer.get_strings(category_idx)

    @staticmethod
    def _reflect_at_boundaries(x: float, low: float = 0.0, high: float = 1.0) -> float:
        """
        Reflects values exceeding boundaries back into a valid range.

        Parameters
        ----------
        x : float
            Input value.
        low : float, optional
            Lower boundary (default is 0.0).
        high : float, optional
            Upper boundary (default is 1.0).

        Returns
        -------
        float
            Value reflected into the valid range.
        """
        while x < low or x > high:
            if x > high:
                x = high - (x - high) / 2.0
            elif x < low:
                x = low + (low - x) / 2.0

        return x

    def _sample_value(self, low: float, high: float, log: bool) -> float:
        """
        Samples a numerical value within the specified range.

        This function generates a random value between `low` and `high`. If `log` is True,
        the sampling is done in logarithmic space, ensuring a proper distribution
        when dealing with exponentially scaled parameters.

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
            return np.exp(self.rng.uniform(np.log(low), np.log(high)))
        else:
            return self.rng.uniform(low, high)

    def optimize(
        self, objective_function: Callable[[Trial], float], n_trials: int
    ) -> tuple:
        """
        Runs the optimization loop.

        Parameters
        ----------
        objective_function : Callable[[Trial], float]
            The function to optimize.
        n_trials : int
            The number of trials.

        Returns
        -------
        tuple
            Best parameters found and the corresponding objective value.
        """
        if not isinstance(n_trials, int):
            raise TypeError("n_trials must be an integer.")

        if n_trials <= 0:
            raise ValueError("n_trials must be a positive integer.")

        if not callable(objective_function):
            raise TypeError("objective_function must be a callable function.")

        self._best_value = float("inf")

        if self.n_init_points is None:
            self.n_init_points = round(np.sqrt(n_trials))
    
        if self.final_noise is None:
            self.final_noise = 1.0 / n_trials
            

        self.n_trials = n_trials
        self.objective_values = np.empty(shape=(n_trials,), dtype=np.float64)
        self.trial_times = np.empty(shape=(n_trials,), dtype=np.float64)

        best_iteration: int = None
        elite_scale: float = 2.0 * np.sqrt(n_trials)
        direction_multipler = 1.0 if self.direction == "minimize" else -1.0

        for iteration in range(self.n_trials):
            start_time = perf_counter()

            if iteration >= self.n_init_points:
                self.progress = iteration / self.n_trials
                self._current_n_elites = max(
                    1, round(elite_scale * self.progress * (1 - self.progress))
                )

                cos_anneal = (1 + np.cos(np.pi * self.progress)) * 0.5

                self._current_noise = (
                    self.final_noise
                    + (self.initial_noise - self.final_noise) * cos_anneal
                )

                self._obj_arg_sort = np.argsort(
                    direction_multipler * self.objective_values[:iteration]
                )
                self._current_cat_temp = 1.0 / (
                    self.final_noise + (1.0 - self.final_noise) * cos_anneal
                )

            self.current_trial = Trial(self, iteration)
            obj_value: float = objective_function(self.current_trial)

            self.trial_times[iteration] = perf_counter() - start_time

            self.objective_values[iteration] = obj_value

            if obj_value < self.best_value:
                self._best_value = obj_value
                best_iteration = iteration

            if self.verbose:
                self._logger.log_trial(
                    iteration=iteration,
                    params=self.current_trial.params,
                    objective=obj_value,
                    time=self.trial_times[iteration],
                    best_value=self._best_value,
                    best_iteration=best_iteration,
                )

        return self

    @staticmethod
    def _validate_init_params(
        n_init_points: Any,
        random_state: Any,
        initial_noise: Any,
        final_noise: Any,
        direction: Any,
        verbose: Any,
    ) -> None:
        """
        Validates initialization parameters for MARSOpt.

        Parameters
        ----------
        n_init_points : Any
            Number of initial random points
        random_state : Any
            Random seed value
        initial_noise : Any
            Initial noise level
        final_noise : Any
            Final noise level
        direction : Any
            Optimization direction ('minimize' or 'maximize')
        verbose : Any
            Verbosity flag

        Raises
        ------
        TypeError
            If parameters are of wrong type
        ValueError
            If parameters have invalid values
        """
        # n_init_points validation
        if n_init_points is not None:
            if not isinstance(n_init_points, int):
                raise TypeError(
                    f"n_init_points must be an integer, got {type(n_init_points)}"
                )
            if n_init_points <= 0:
                raise ValueError(f"n_init_points must be positive, got {n_init_points}")

        # random_state validation
        if random_state is not None and not isinstance(random_state, int):
            raise TypeError(
                f"random_state must be None or an integer, got {type(random_state)}"
            )

        # initial_noise validation
        if not isinstance(initial_noise, (int, float)):
            raise TypeError(f"initial_noise must be a number, got {type(initial_noise)}")
        if not (0 < initial_noise <= 1):
            raise ValueError(
                f"initial_noise must be between 0 and 1 (exclusive), got {initial_noise}"
            )

        # final_noise validation
        if final_noise is not None:
            if not isinstance(final_noise, (int, float)):
                raise TypeError(f"final_noise must be a number, got {type(final_noise)}")
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
    def best_value(self) -> float:
        return self._best_value

    @property
    def best_trial(self) -> dict:
        """
        Returns the best trial's parameters and iteration number.

        Returns
        -------
        dict
            Dictionary containing the iteration number and parameter values of the best trial.
        """
        best_iteration = int(self._obj_arg_sort[0])

        # Pre-allocate dictionary with known size
        best_trial_dict = {
            "iteration": best_iteration,
            **{
                param_name: (
                    int(param.values[best_iteration])
                    if param.type == int
                    else (
                        float(param.values[best_iteration])
                        if param.type == float
                        else param.category_indexer.get_strings(
                            np.argmax(param.values[best_iteration])
                        )
                    )
                )
                for param_name, param in self.parameters.items()
            },
        }

        return best_trial_dict

    @property
    def trial_history(self) -> List[dict]:
        """
        Returns the complete history of all trials as a list of dictionaries.
        Each dictionary represents one trial iteration with its parameters and results.

        Returns
        -------
        List[dict]
            List of dictionaries where each dictionary contains:
            - iteration: Trial iteration number
            - objective_value: Objective function value
            - trial_time: Execution time of the trial
            - parameters: Dictionary of parameter values for that trial
        """
        final_iteration = min(
            int((self.progress * self.n_trials) + 1), len(self.objective_values)
        )
        history = []

        for iteration in range(final_iteration):
            trial_dict = {
                "iteration": iteration,
                "objective_value": float(self.objective_values[iteration]),
                "trial_time": float(self.trial_times[iteration]),
                "parameters": {},
            }

            # Add parameter values for this iteration
            for param_name, param in self.parameters.items():
                if param.type == int:
                    value = int(param.values[iteration])
                elif param.type == float:
                    value = float(param.values[iteration])
                else:  # categorical parameters
                    value = param.category_indexer.get_strings(
                        np.argmax(param.values[iteration])
                    )
                trial_dict["parameters"][param_name] = value

            history.append(trial_dict)

        return history
