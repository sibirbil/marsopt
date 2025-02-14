import numpy as np
from numpy.typing import NDArray
from typing import List, Union, Dict


class CategoryIndexer:
    __slots__ = ["str_to_idx", "idx_to_str", "next_idx"]
    
    """
    A helper class for managing categorical parameter indexing.

    Attributes
    ----------
    str_to_idx : Dict[str, int]
        A dictionary mapping category names to unique integer indices.
    idx_to_str : Dict[int, str]
        A dictionary mapping integer indices back to category names.
    next_idx : int
        The next available index for a new category.
    """

    def __init__(self) -> None:
        """
        Initializes the category indexer with bidirectional mappings.
        """
        self.str_to_idx: Dict[str, int] = {}
        self.idx_to_str: Dict[int, str] = {}
        self.next_idx: int = 0

    def get_indices(self, strings: List[str]) -> NDArray:
        """
        Efficiently converts a list of category names to their corresponding integer indices.

        Parameters
        ----------
        strings : List[str]
            List of category names.

        Returns
        -------
        NDArray
            Array of corresponding integer indices.
        """
        indices = np.empty(len(strings), dtype=np.int32)
        for i, s in enumerate(strings):
            if s not in self.str_to_idx:
                self.str_to_idx[s] = self.next_idx
                self.idx_to_str[self.next_idx] = s
                self.next_idx += 1
            indices[i] = self.str_to_idx[s]
        return indices

    def get_strings(self, indice: int) -> str:
        """
        Efficiently converts an integer index to its corresponding category name.

        Parameters
        ----------
        indice : int
            Category index.

        Returns
        -------
        str
            Corresponding category name.
        """
        return self.idx_to_str[indice]

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

class Parameter:
    __slots__ = ["name", "type", "values", "category_indexer"]

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.type: type = None
        self.values: NDArray = None
        self.category_indexer = CategoryIndexer()

    def set_values(
        self, max_iter: int, param_type_or_categories: Union[int, float, List[str]]
    ) -> None:
        """
        Initializes or updates the storage for parameter values based on the parameter type.

        Parameters
        ----------
        max_iter : int
            Maximum number of iterations the parameter will be used for.
        param_type_or_categories : Union[type, List[str]]
            Either a type (int, float) or a list of categorical values.

        Raises
        ------
        ValueError
            If an unsupported parameter type is provided.
        """
        # Handle numeric types
        if isinstance(param_type_or_categories, type):
            self.values = np.empty(shape=(max_iter,), dtype=np.float64)
            self.type = param_type_or_categories
            return

        # Handle categorical types
        if isinstance(param_type_or_categories, list):
            categories = param_type_or_categories
            if not categories:
                raise ValueError("Categories list cannot be empty")

            # Get indices for all categories
            category_indices = self.category_indexer.get_indices(categories)
            required_width = category_indices.max() + 1

            # Initialize or resize values array as needed
            if self.values is None:
                self.values = np.zeros((max_iter, required_width), dtype=np.float64)
                self.type = list
            else:
                current_width = self.values.shape[1]
                if required_width > current_width:
                    # Efficiently extend the array only if needed
                    extension = np.zeros((max_iter, required_width - current_width), dtype=np.float64)
                    self.values = np.hstack((self.values, extension))
            return
