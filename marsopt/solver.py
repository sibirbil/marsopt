from typing import Dict, List, Any, Optional
import numpy as np
from numpy.typing import NDArray
from .parameters import Parameter


class Trial:
    def __init__(self, optimizer: "MARSOpt", trial_id: int):
        self.optimizer = optimizer
        self.trial_id = trial_id
        self.params: Dict[str, Any] = {}

    def suggest_float(
        self, name: str, low: float, high: float, log: bool = False
    ) -> float:
        """Suggest a float parameter value"""
        value = self.optimizer._suggest_numerical(name, low, high, float, log)
        self.params[name] = value
        return value

    def suggest_int(self, name: str, low: int, high: int, log: bool = False) -> int:
        """Suggest an integer parameter value"""
        value = self.optimizer._suggest_numerical(name, low, high, int, log)
        self.params[name] = value
        return value

    def suggest_categorical(self, name: str, categories: List[Any]) -> Any:
        """Suggest a categorical parameter value"""
        value = self.optimizer._suggest_categorical(name, categories)
        self.params[name] = value
        return value


class MARSOpt:
    """Main optimizer working directly in parameter space"""

    def __init__(
        self,
        n_init_points: int = 10,
        random_state: Optional[int] = None,
        initial_noise: float = 0.20,
        min_temperature: float = 0.20,
    ):
        self.n_init_points = n_init_points
        self.initial_noise = initial_noise
        self.min_temperature = min_temperature
        self.rng = np.random.default_rng(random_state)

        self.parameters: Dict[str, Parameter] = {}
        self.objective_values: NDArray = None
        self.current_trial: Optional[Trial] = None
        
        self._progress: float = None
        self._current_noise: float = None
        self._current_n_elites: float = None
        self._obj_arg_sort: NDArray = None

    def _suggest_numerical(
        self, name: str, low: float, high: float, param_type: str, log: bool
    ) -> float:
        param = self.parameters.get(name)

        if param is None:
            param = Parameter(
                name=name,
            )
            param.set_values(
                max_iter=self.max_iter, param_type_or_categories=param_type
            )
            self.parameters[name] = param

        if self.current_trial.trial_id < self.n_init_points:
            if log:
                log_low = np.log(low)
                log_high = np.log(high)
                value = np.exp(self.rng.uniform(log_low, log_high))
            else:
                value = self.rng.uniform(low, high)

        else:
            param_values = param.values[: self.current_trial.trial_id]
            range_mask = (param_values >= low) & (param_values <= high)

            if not np.any(range_mask):
                if log:
                    log_low = np.log(low)
                    log_high = np.log(high)
                    value = np.exp(self.rng.uniform(log_low, log_high))
                else:
                    value = self.rng.uniform(low, high)

            else:
                objective_masked = self.objective_values[: self.current_trial.trial_id][
                    range_mask
                ]
                best_objectives = np.argsort(objective_masked)
                values_masked = param_values[range_mask][best_objectives]
                base_value = self.rng.choice(values_masked[: self._current_n_elites])

                if log:
                    log_base = np.log(base_value)
                    log_range = np.log(high) - np.log(low)
                    noise = self.rng.normal(
                        loc=0.0, scale=self._current_noise * log_range
                    )
                    value = np.exp(log_base + noise)
                else:
                    # Apply noise directly
                    param_range = high - low
                    noise = self.rng.normal(
                        loc=0.0, scale=self._current_noise * param_range
                    )
                    value = base_value + noise

                # Clip to bounds
                value = self.reflect_at_boundaries(value, low, high)

            if param_type == int:
                # probabilistic rounding
                frac = value - np.floor(value)
                if self.rng.random() < frac:
                    value = int(np.ceil(value))
                else:
                    value = int(np.floor(value))

        param.values[self.current_trial.trial_id] = value

        return value

    def _suggest_categorical(self, name: str, categories: List[Any]) -> Any:
        """Handle categorical parameter suggestions"""
        param = self.parameters.get(name)

        if param is None:
            param = Parameter(name=name)
            param.set_values(
                max_iter=self.max_iter, param_type_or_categories=categories
            )
            self.parameters[name] = param

        cat_indices = param.category_indexer.get_indices(categories)

        if self.current_trial.trial_id < self.n_init_points:
            category_idx = self.rng.choice(cat_indices)
        else:
            best_objectives = np.argsort(
                self.objective_values[: self.current_trial.trial_id]
            )[: self._current_n_elites]
            param_values = param_values[best_objectives, cat_indices]

            noise = self.rng.normal(loc=0.0, scale=self._current_noise)

            chosen_elites = self.rng.choice(
                self._current_n_elites, size=len(cat_indices), replace=True
            )

            chosen_elites_with_noise = param_values[chosen_elites, cat_indices] + noise

            for i in range(chosen_elites_with_noise.size):
                chosen_elites_with_noise[i] = self.reflect_at_boundaries(
                    chosen_elites_with_noise[i]
                )

            temp = max(
                self.min_temperature,
                self.progress,
            )

            exps = np.exp(
                chosen_elites_with_noise / temp
                - np.max(chosen_elites_with_noise) / temp
            )
            probs = exps / exps.sum()

            category_idx = cat_indices[self.rng.choice(len(probs), p=probs)]

        result = np.zeros(len(param.category_indexer), dtype=np.float64)
        result[category_idx] = 1.0

        param.values[self.current_trial.trial_id, :] = result

        return param.category_indexer.get_strings(category_idx)

    @staticmethod
    def reflect_at_boundaries(x: float, low: float = 0.0, high: float = 1.0) -> float:
        """
        Reflects values that exceed boundaries [low, high] back into the valid range.
        For values > high: reflects back half of the excess
        For values < low: reflects back half of the deficit
        
        Args:
            x: Input value that may exceed [low, high] bounds
            low: Lower boundary of valid range
            high: Upper boundary of valid range
        
        Returns:
            Float value reflected back into [low, high] range
        """
        if x > high:
            excess = x - high
            return high - (excess / 2.0)
        elif x < low:
            deficit = low - x
            return low + (deficit / 2.0)
        else:
            return x

    def objective_wrapper(self, objective_function, iteration):
        self.current_trial = Trial(self, iteration)
        obj_value = objective_function(self.current_trial)
        return obj_value

    def optimize(self, objective_function: callable, n_trial: int ):
        """Run optimization loop"""
        best_value = float("inf")
        best_params = None
        
        self.max_iter = n_trial
        self.final_noise = 1.0 / n_trial
        self.objective_values = np.empty(shape=(n_trial,), dtype=np.float64)
        self._elite_scale: float = 2 * np.sqrt(n_trial)
        
        for iteration in range(self.max_iter):
            if iteration >= self.n_init_points:
                self.progress = iteration / self.max_iter
                self._current_n_elites = max(
                    1, round(self._elite_scale * self.progress * (1 - self.progress))
                )
                self._current_noise = self.final_noise + 0.5 * (
                    self.initial_noise - self.final_noise
                ) * (1 + np.cos(np.pi * self.progress))

            # self._obj_arg_sort = np.argsort(obj_value[: iteration + 1])
            obj_value = self.objective_wrapper(objective_function, iteration)

            self.objective_values[iteration] = obj_value

            if obj_value < best_value:
                best_value = obj_value
                best_params = self.current_trial.params.copy()

        return best_params, best_value
