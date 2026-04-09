from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
from numpy.typing import NDArray

class Trial:
    """
    Represents a single trial in the optimization process.
    """

    study: Study
    trial_id: int
    variables: Dict[str, Any]
    user_attrs: Dict[str, Any]

    def __init__(self, study: Study, trial_id: int) -> None:
        """
        Initialize a Trial instance.

        Parameters
        ----------
        study : Study
            The study associated with this trial.
        trial_id : int
            The unique identifier for this trial.
        """
        ...

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
        ...

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
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
        ...

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
        ...

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
        ...

class Study:
    """
    Mixed Adaptive Random Search for Optimization
    """

    n_trials: int
    n_init_points: Optional[int]
    initial_noise: float
    final_noise: Optional[float]
    verbose: bool
    direction: str
    epsilon: float
    elite_window: Optional[int]

    def __init__(
        self,
        initial_noise: float = 0.33,
        direction: str = "minimize",
        n_init_points: Optional[int] = None,
        final_noise: Optional[float] = None,
        epsilon: float = 1.0,
        elite_window: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
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
            Seed for reproducibility.
        verbose : bool, default = True
            Whether to print logs during optimization.
        """
        ...

    def optimize(
        self,
        objective_function: Callable[[Trial], Union[float, int]],
        n_trials: int,
    ) -> None:
        """
        Runs the optimization loop.

        Parameters
        ----------
        objective_function : Callable[[Trial], Union[float, int]]
            The function to optimize.
        n_trials : int
            The number of trials.
        """
        ...

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
            - **variables** (:obj:`Dict[str, Union[int, float, str]]`)
              A dictionary of variable values from the best trial.
            - **user_attrs** (:obj:`Dict[str, Any]`)
              A dictionary of user-defined attributes for the best trial.
        """
        ...

    @property
    def trials(self) -> List[Dict[str, Any]]:
        """
        Get the complete history of all trials in the optimization process.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries, where each dictionary represents a trial
            with keys: iteration, objective_value, trial_time, variables, user_attrs.
        """
        ...

    @property
    def objective_values(self) -> NDArray[np.float64]:
        """
        Returns the objective function values for all completed trials.

        Returns
        -------
        NDArray[np.float64]
            A NumPy array containing the recorded objective function values.
        """
        ...

    @property
    def elapsed_times(self) -> NDArray[np.float64]:
        """
        Returns the execution times of all completed trials.

        Returns
        -------
        NDArray[np.float64]
            A NumPy array containing the recorded execution times (in seconds).
        """
        ...
