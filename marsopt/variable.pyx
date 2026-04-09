# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp

cnp.import_array()


cdef class CategoryIndexer:
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

    def __init__(self):
        """Initializes the category indexer with bidirectional mappings."""
        self.str_to_idx = {}
        self.idx_to_str = {}
        self.next_idx = 0

    cpdef cnp.ndarray get_indices(self, list strings):
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
        cdef int n = len(strings)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] indices = np.empty(n, dtype=np.int32)
        cdef int i
        cdef str s
        cdef int idx
        for i in range(n):
            s = strings[i]
            if s not in self.str_to_idx:
                idx = self.next_idx
                self.str_to_idx[s] = idx
                self.idx_to_str[idx] = s
                self.next_idx = idx + 1
            indices[i] = <cnp.int32_t>self.str_to_idx[s]
        return indices

    cpdef str get_strings(self, int indice):
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
        return <str>self.idx_to_str[indice]

    def __len__(self):
        """Returns the number of unique categories."""
        return len(self.str_to_idx)


cdef class Variable:
    """Represents an optimization variable."""

    def __init__(self, str name):
        self.name = name
        self.type = None
        self.values = None
        self.category_indexer = CategoryIndexer()

    def set_values(self, int max_iter, var_type_or_categories):
        """
        Initializes or updates the storage for variable values based on the variable type.

        Parameters
        ----------
        max_iter : int
            Maximum number of iterations the variable will be used for.
        var_type_or_categories : Union[type, List[str]]
            Either a type (int, float) or a list of categorical values.
        """
        if isinstance(var_type_or_categories, type):
            self.values = np.full(max_iter, fill_value=np.nan, dtype=np.float64)
            self.type = var_type_or_categories
            return

        if isinstance(var_type_or_categories, list):
            categories = var_type_or_categories
            if not categories:
                raise ValueError("Categories list cannot be empty")
            self.category_indexer.get_indices(categories)
            if self.values is None:
                self.values = np.full(max_iter, fill_value=-1, dtype=np.int32)
                self.type = list
            return

    def add_iter(self, int additional_iter):
        """
        Adds additional iterations by extending the values array.

        Parameters
        ----------
        additional_iter : int
            The number of additional iterations to add.
        """
        if additional_iter <= 0:
            raise ValueError("additional_iter must be greater than zero.")
        if self.values is None:
            raise ValueError("Values array has not been initialized.")

        if self.type is list:
            extension = np.full(additional_iter, fill_value=-1, dtype=np.int32)
            self.values = np.concatenate((self.values, extension))
        else:
            extension = np.full(additional_iter, fill_value=np.nan, dtype=self.values.dtype)
            self.values = np.concatenate((self.values, extension))
