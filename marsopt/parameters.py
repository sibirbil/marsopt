import numpy as np
from numpy.typing import NDArray
from typing import List, Union


class CategoryIndexer:
    """
    A helper class for managing categorical parameter indexing.

    Attributes
    ----------
    str_to_idx : dict
        A dictionary mapping category names (strings) to unique integer indices.
    next_idx : int
        The next available index for a new category.
    """

    def __init__(self) -> None:
        """
        Initializes the category indexer.
        """
        self.str_to_idx = {}
        self.next_idx = 0

    def get_indices(self, strings: List[str]) -> NDArray:
        """
        Converts a list of category names to their corresponding integer indices.

        Parameters
        ----------
        strings : List[str]
            List of category names.

        Returns
        -------
        List[int]
            List of corresponding integer indices.
        """
        result = []
        for s in strings:
            if s not in self.str_to_idx:
                self.str_to_idx[s] = self.next_idx
                self.next_idx += 1
            result.append(self.str_to_idx[s])
        return np.array(result)

    def get_strings(self, indice: int) -> str:
        """
        Converts a list of integer indices back to their corresponding category names.

        Parameters
        ----------
        indices : List[int]
            List of category indices.

        Returns
        -------
        List[str]
            List of corresponding category names.
        """
        idx_to_str = {v: k for k, v in self.str_to_idx.items()}
        return idx_to_str[indice]

    def __len__(self) -> int:
        """
        Returns the number of unique categories.

        Returns
        -------
        int
            Number of categories stored in the indexer.
        """
        return len(self.str_to_idx)


class Parameter:
    """
    Represents a parameter in an optimization process.

    Attributes
    ----------
    name : str
        The name of the parameter.
    type : type
        The type of the parameter (int, float, or categorical).
    values : NDArray
        An array storing parameter values.
    category_indexer : CategoryIndexer
        Manages category indices for categorical parameters.
    """

    __slots__ = ["name", "type", "values", "category_indexer"]

    def __init__(self, name: str) -> None:
        """
        Initializes a Parameter instance.

        Parameters
        ----------
        name : str
            The name of the parameter.
        """
        self.name: str = name
        self.type: type = None
        self.values: NDArray = None
        self.category_indexer = CategoryIndexer()

    def __repr__(self) -> str:
        """
        Returns a string representation of the Parameter instance.

        Returns
        -------
        str
            String representation including the name, type, and value preview.
        """
        values_info = ""
        if self.values is not None:
            if self.values.size > 0:
                # Show first few values if they exist
                preview = np.array2string(
                    self.values.flatten()[:3], precision=2, separator=", "
                )[:-1]
                values_info = f", values_preview={preview}...]"
            else:
                values_info = ", values=empty"

        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"type={self.type.__name__ if self.type else 'None'}"
            f"{values_info})"
        )

    def set_values(
        self, max_iter: int, param_type_or_categories: Union[int, float, List[str]]
    ) -> None:
        """
        Initializes the storage for parameter values based on the parameter type.

        Parameters
        ----------
        max_iter : int
            Maximum number of iterations the parameter will be used for.
        param_type_or_categories : Union[int, float, List[str]]
            The type of parameter (int, float) or a list of categorical values.

        Raises
        ------
        ValueError
            If an unsupported parameter type is provided.
        """
        if isinstance(param_type_or_categories, type):
            if param_type_or_categories == int:
                self.values = np.empty(shape=(max_iter,), dtype=np.float64)
                self.type = int
            elif param_type_or_categories == float:
                self.values = np.empty(shape=(max_iter,), dtype=np.float64)
                self.type = float
            else:
                raise ValueError(f"Unsupported type: {param_type_or_categories}")
        elif isinstance(param_type_or_categories, list):
            categories = param_type_or_categories
            total_cats = len(self.category_indexer)
            max_indice = max(self.category_indexer.get_indices(categories))

            if total_cats == 0:
                self.values = np.empty(shape=(max_iter, max_indice + 1), dtype=np.float64)
                self.type = type(categories)
            elif total_cats < max_indice:
                self.values = np.vstack(
                    (self.values, np.zeros(shape=(max_iter, 1), dtype=np.float64))
                )
        else:
            raise ValueError(
                f"Unsupported parameter type: {type(param_type_or_categories)}"
            )
