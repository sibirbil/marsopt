from typing import Dict, List, Optional, Union
import numpy as np
from numpy.typing import NDArray

class CategoryIndexer:
    """
    A helper class for managing categorical variable indexing.

    Attributes
    ----------
    str_to_idx : Dict[str, int]
        A dictionary mapping category names to unique integer indices.
    idx_to_str : Dict[int, str]
        A dictionary mapping integer indices back to category names.
    next_idx : int
        The next available index for a new category.
    """

    str_to_idx: Dict[str, int]
    idx_to_str: Dict[int, str]
    next_idx: int

    def __init__(self) -> None:
        """Initializes the category indexer with bidirectional mappings."""
        ...

    def get_indices(self, strings: List[str]) -> NDArray[np.int32]:
        """
        Converts a list of category names to their corresponding integer indices.

        Parameters
        ----------
        strings : List[str]
            List of category names.

        Returns
        -------
        NDArray[np.int32]
            Array of corresponding integer indices.
        """
        ...

    def get_strings(self, indice: int) -> str:
        """
        Converts an integer index to its corresponding category name.

        Parameters
        ----------
        indice : int
            Category index.

        Returns
        -------
        str
            Corresponding category name.
        """
        ...

    def __len__(self) -> int:
        """
        Returns the number of unique categories.

        Returns
        -------
        int
            Number of categories stored in the indexer.
        """
        ...

class Variable:
    """
    Represents an optimization variable.
    """

    name: str
    type: Optional[type]
    values: Optional[NDArray]
    category_indexer: CategoryIndexer

    def __init__(self, name: str) -> None: ...

    def set_values(self, max_iter: int, var_type_or_categories: Union[type, List[str]]) -> None:
        """
        Initializes or updates the storage for variable values based on the variable type.

        Parameters
        ----------
        max_iter : int
            Maximum number of iterations the variable will be used for.
        var_type_or_categories : Union[type, List[str]]
            Either a type (int, float) or a list of categorical values.
        """
        ...

    def add_iter(self, additional_iter: int) -> None:
        """
        Adds additional iterations by extending the values array.

        Parameters
        ----------
        additional_iter : int
            The number of additional iterations to add.

        Raises
        ------
        ValueError
            If additional_iter is less than or equal to zero.
        """
        ...
