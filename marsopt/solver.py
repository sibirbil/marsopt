from typing import Dict, List, Any, Optional, Callable, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from time import perf_counter

from .parameters import Parameter
from .logger import OptimizationLogger

from functools import lru_cache


class Trial:
    __slots__ = ["study", "trial_id", "params", "_validated_params"]

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
        self.params: Dict[str, Any] = {}
        self._validated_params = set()

    def __repr__(self) -> str:
        """
        Return a string representation of the Trial instance.

        Returns
        -------
        str
            A string representation of the Trial instance.
        """
        return f"Trial(trial_id={self.trial_id}, params={self.params})"

    @staticmethod
    @lru_cache(maxsize=None)
    def _validate_numerical_cached(
        name: str, low: Any, high: Any, expected_type: type, log: bool
    ) -> None:
        """
        Validate numerical parameters using cached results.

        Parameters
        ----------
        name : str
            The name of the parameter.
        low : Any
            The lower bound of the parameter range.
        high : Any
            The upper bound of the parameter range.
        expected_type : type
            The expected type of the parameter (int or float).
        log : bool
            Whether the parameter is log-scaled.
        """
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
        """
        Validate numerical parameters and cache the results.

        Parameters
        ----------
        name : str
            The name of the parameter.
        low : Any
            The lower bound of the parameter range.
        high : Any
            The upper bound of the parameter range.
        expected_type : type
            The expected type of the parameter (int or float).
        log : bool
            Whether the parameter is log-scaled.
        """
        if not isinstance(name, str):
            raise TypeError(f"Parameter name must be a string, got {type(name)}")

        if name == "":
            raise ValueError("Parameter name cannot be an empty string.")

        self._validate_numerical_cached(name, low, high, expected_type, log)
        self._validated_params.add(name)

    @staticmethod
    @lru_cache(maxsize=None)
    def _validate_categorical_cached(
        name: str, categories_tuple: Tuple[Any, ...]
    ) -> None:
        """
        Validate categorical parameters using cached results.

        Parameters
        ----------
        name : str
            The name of the parameter.
        categories_tuple : Tuple[Any, ...]
            A tuple of valid categorical values.
        """
        if len(categories_tuple) < 1:
            raise ValueError(
                f"Parameter '{name}': 'categories' must contain at least one element"
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
        """
        Validate categorical parameters and cache the results.

        Parameters
        ----------
        name : str
            The name of the parameter.
        categories : List[Any]
            A list of valid categorical values.
        """
        if not isinstance(name, str):
            raise TypeError(f"Parameter name must be a string, got {type(name)}")

        if name == "":
            raise ValueError("Parameter name cannot be an empty string.")

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
        Suggest a floating-point parameter value.

        Parameters
        ----------
        name : str
            The name of the parameter.
        low : float
            The lower bound of the parameter range.
        high : float
            The upper bound of the parameter range.
        log : bool, optional
            Whether the parameter is log-scaled (default is False).

        Returns
        -------
        float
            The suggested floating-point value.
        """
        self._validate_numerical(name, low, high, float, log)
        value = self.study._suggest_numerical(name, low, high, float, log)
        self.params[name] = value
        return value

    def suggest_int(self, name: str, low: int, high: int, log: bool = False) -> int:
        """
        Suggest an integer parameter value.

        Parameters
        ----------
        name : str
            The name of the parameter.
        low : int
            The lower bound of the parameter range.
        high : int
            The upper bound of the parameter range.
        log : bool, optional
            Whether the parameter is log-scaled (default is False).

        Returns
        -------
        int
            The suggested integer value.
        """
        self._validate_numerical(name, low, high, int, log)
        value = self.study._suggest_numerical(name, low, high, int, log)
        self.params[name] = value
        return value

    def suggest_categorical(self, name: str, categories: List[Any]) -> Any:
        """
        Suggest a categorical parameter value.

        Parameters
        ----------
        name : str
            The name of the parameter.
        categories : List[Any]
            A list of valid categorical values.

        Returns
        -------
        Any
            The suggested categorical value.
        """
        self._validate_categorical(name, categories)
        value = self.study._suggest_categorical(name, categories)
        self.params[name] = value
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
            Direction of optimization, either **"minimize"** or **"maximize"**.
        n_init_points : int, default = None
            Number of initial random points. If `None`, it is set as:
            **round(sqrt(`n_trials`))**
        final_noise : float, default = None
            Final noise level. If `None`, it is set as:
            **1.0 / `n_trials`**
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
        )

        self.n_init_points = n_init_points
        self.initial_noise = initial_noise
        self.verbose = verbose
        self.direction = direction
        self.final_noise = final_noise

        self._rng = np.random.RandomState(random_state)
        self._objective_values: NDArray[np.float64] = None
        self._elapsed_times: NDArray[np.float64] = None
        self._current_trial: Optional[Trial] = None
        self._parameters: Dict[str, Parameter] = {}

        self._progress: float = None
        self._current_noise: float = None
        self._current_n_elites: float = None
        self._current_cat_temp: float = None
        self._obj_arg_sort: NDArray[np.int64] = None
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
        self, name: str, low: float, high: float, param_type: type, log: bool
    ) -> Union[float, int]:
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
        Union[float, int]
            The suggested numerical value.
        """
        param = self._parameters.get(name)

        if param is None:
            param = Parameter(
                name=name,
            )
            param.set_values(
                max_iter=self.n_trials, param_type_or_categories=param_type
            )
            self._parameters[name] = param
            
        else:
            if param.type != param_type:
                raise TypeError(
                    f"Parameter '{name}' has already been registered with type {param.type}, "
                    f"but an attempt was made to register it as type {param_type}. Ensure consistency."
                )


        if self._current_trial.trial_id < self.n_init_points:
            value = self._sample_value(low, high, log)

        else:
            param_values = param.values[: self._current_trial.trial_id]
            range_mask = (param_values >= low) & (param_values <= high)

            if not np.any(range_mask):
                value = self._sample_value(low, high, log)

            else:
                sorted_indices = self._obj_arg_sort[range_mask[self._obj_arg_sort]]
                values_masked = param_values[sorted_indices]
                base_value = self._rng.choice(values_masked[: self._current_n_elites])

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
                    # Apply noise directly
                    param_range = high - low
                    noise = self._rng.normal(
                        loc=0.0, scale=self._current_noise * param_range
                    )

                    value = self._reflect_at_boundaries(base_value + noise, low, high)

        if param_type == int:
            # probabilistic rounding
            frac = value - np.floor(value)
            if self._rng.random() < frac:
                value = int(np.ceil(value))
            else:
                value = int(np.floor(value))

        param.values[self._current_trial.trial_id] = value

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
        param = self._parameters.get(name)
        trial_id = self._current_trial.trial_id

        if param is None:
            param = Parameter(name=name)
            param.set_values(
                max_iter=self.n_trials, param_type_or_categories=categories
            )
            self._parameters[name] = param

        else:
            if param.type != type(categories):
                raise TypeError(
                    f"Parameter '{name}' has already been registered with type {param.type}, "
                    f"but an attempt was made to register it as type {type(categories)}. Ensure consistency."
                )
            
            
            param.set_values(
                max_iter=self.n_trials, param_type_or_categories=categories
            )

        cat_indices = param.category_indexer.get_indices(categories)
        cat_size = cat_indices.size

        if trial_id < self.n_init_points:
            category_idx = self._rng.choice(cat_indices)

        else:
            sorted_trials = self._obj_arg_sort[: self._current_n_elites]

            # Get parameter values for the best trials
            param_values = param.values[sorted_trials[:, np.newaxis], cat_indices]

            noise = self._rng.normal(loc=0.0, scale=self._current_noise, size=cat_size)

            chosen_elites_with_noise = param_values.mean(axis=0) + noise

            for i in range(cat_size):
                chosen_elites_with_noise[i] = self._reflect_at_boundaries(
                    chosen_elites_with_noise[i]
                )

            exps = np.exp(
                (chosen_elites_with_noise - np.max(chosen_elites_with_noise))
                * self._current_cat_temp
            )
            probs = exps / exps.sum()

            category_idx = cat_indices[self._rng.choice(cat_size, p=probs)]

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
        low : float, optional, default = 0.0
            Lower boundary.
        high : float, optional, default = 1.0
            Upper boundary.

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
            return np.exp(self._rng.uniform(np.log(low), np.log(high)))
        else:
            return self._rng.uniform(low, high)

    def optimize(
        self, objective_function: Callable[[Trial], float], n_trials: int
    ) -> None:
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
        None
        """
        if not isinstance(n_trials, int):
            raise TypeError("n_trials must be an integer.")

        if n_trials <= 0:
            raise ValueError("n_trials must be a positive integer.")

        if not callable(objective_function):
            raise TypeError("objective_function must be a callable function.")

        ## check existing trial:
        if self._objective_values is not None:
            n_exist_trials = int(self._objective_values.size)

            # Find best iteration based on direction
            if self.direction == "minimize":
                best_iteration = np.argmin(self._objective_values[:n_exist_trials])
                best_value = self._objective_values[best_iteration]
            else:
                best_iteration = np.argmax(self._objective_values[:n_exist_trials])
                best_value = self._objective_values[best_iteration]

            total_trials = n_trials + n_exist_trials
            self.n_trials = total_trials

            if self.final_noise is None:
                self.final_noise = 1.0 / total_trials

            elite_scale: float = 2.0 * np.sqrt(total_trials)

            # Correctly use np.hstack with tuples
            old_objective_values = self._objective_values
            old_elapsed_times = self._elapsed_times

            self._objective_values = np.empty(shape=(total_trials,), dtype=np.float64)
            self._elapsed_times = np.empty(shape=(total_trials,), dtype=np.float64)

            # Copy existing data
            self._objective_values[:n_exist_trials] = old_objective_values
            self._elapsed_times[:n_exist_trials] = old_elapsed_times

            for param in self._parameters.keys():
                self._parameters[param].add_iter(n_trials)

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
                self.final_noise = 1.0 / n_trials

            self.n_trials = n_trials
            self._objective_values = np.empty(shape=(n_trials,), dtype=np.float64)
            self._elapsed_times = np.empty(shape=(n_trials,), dtype=np.float64)

            elite_scale: float = 2.0 * np.sqrt(n_trials)

            if self.n_init_points is None:
                self.n_init_points = round(np.sqrt(self.n_trials))

        direction_multipler = 1.0 if self.direction == "minimize" else -1.0

        # Start from the existing trials count
        for iteration in range(n_exist_trials, total_trials):
            start_time = perf_counter()

            if iteration >= self.n_init_points:
                self._current_n_elites = max(
                    1, round(elite_scale * self.progress * (1 - self.progress))
                )

                cos_anneal = (1 + np.cos(np.pi * self.progress)) * 0.5

                self._current_noise = (
                    self.final_noise
                    + (self.initial_noise - self.final_noise) * cos_anneal
                )

                self._current_cat_temp = 1.0 / (
                    self.final_noise + (1.0 - self.final_noise) * cos_anneal
                )

            self.progress = iteration / self.n_trials

            self._obj_arg_sort = np.argsort(
                direction_multipler * self._objective_values[:iteration]
            )

            self._current_trial = Trial(self, iteration)
            obj_value: float = objective_function(self._current_trial)

            self._elapsed_times[iteration] = perf_counter() - start_time
            self._objective_values[iteration] = obj_value

            # Update best value based on optimization direction
            if (self.direction == "minimize" and obj_value < best_value) or (
                self.direction == "maximize" and obj_value > best_value
            ):
                best_value = obj_value
                best_iteration = iteration

            if self.verbose:
                self._logger.log_trial(
                    iteration=iteration,
                    params=self._current_trial.params,
                    objective=obj_value,
                    best_value=best_value,
                    best_iteration=best_iteration,
                )

        return

    def parameter_importance(self) -> Dict[str, float]:
        """
        Calculates the importance of each parameter based on correlation with objective values.
        Uses Spearman correlation for both numerical and categorical parameters.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping parameter names to their importance scores (absolute correlation values).
            Higher values indicate stronger correlation with the objective.
        """
        try:
            from scipy.stats import spearmanr
        except:
            raise ImportError(
                "ImportError: The 'scipy' library is not found. Please install",
                "it using 'pip install scipy' to compute parameter importance.",
            )

        if self._objective_values is None or len(self._parameters) == 0:
            raise ValueError("No trials have been conducted yet.")

        importances = {}
        completed_trials = min(
            int((self.progress * self.n_trials) + 1), len(self._objective_values)
        )

        objective_values = self._objective_values[:completed_trials]

        # Handle optimization direction
        if self.direction == "maximize":
            objective_values = -objective_values  # Convert to minimization problem

        for param_name, param in self._parameters.items():
            if param.type in (int, float):
                # For numerical parameters, use the raw values
                param_values = param.values[:completed_trials]
            else:
                # For categorical parameters, use the index of the selected category
                param_values = np.argmax(param.values[:completed_trials], axis=1)

            # Calculate Spearman correlation using scipy's implementation
            correlation, _ = spearmanr(param_values, objective_values)

            # Handle NaN values that might occur with constant parameters
            if np.isnan(correlation):
                correlation = 0.0

            importances[param_name] = abs(correlation)

        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

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
        Validates initialization parameters for Study.

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
        objective value, execution time, and parameter values.

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

            - **params** (:obj:`Dict[str, Union[int, float, str]]`)

            A dictionary of parameter values from the best trial. Keys are parameter names,
            and values are their respective values (int, float, or categorical as a string).

        """
        best_iteration = int(self._obj_arg_sort[0])

        return {
            "iteration": best_iteration,
            "objective_value": float(self._objective_values[best_iteration]),
            "trial_time": float(self._elapsed_times[best_iteration]),
            "params": {
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
                for param_name, param in self._parameters.items()
            },
        }

    @property
    def trials(self) -> List[Dict[str, Any]]:
        """
        Get the complete history of all trials in the optimization process.

        Each trial includes its iteration number, objective function value, execution time,
        and parameter values.

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

            - **parameters** (:obj:`Dict[str, Union[int, float, str]]`)

            A dictionary of parameter values from the trial. Keys are parameter names,
            and values are their respective values (int, float, or categorical as a string).
        """
        final_iteration = min(
            int((self.progress * self.n_trials) + 1), len(self._objective_values)
        )
        history = []

        for iteration in range(final_iteration):
            trial_dict = {
                "iteration": iteration,
                "objective_value": float(self._objective_values[iteration]),
                "trial_time": float(self._elapsed_times[iteration]),
                "parameters": {},
            }

            # Store parameter values for this iteration
            for param_name, param in self._parameters.items():
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
        return self._objective_values

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
        return self._elapsed_times
