# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp

cnp.import_array()


cdef class CategoryIndexer:

    def __init__(self):
        self.str_to_idx = {}
        self.idx_to_str = {}
        self.next_idx = 0

    cpdef cnp.ndarray get_indices(self, list strings):
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
        return <str>self.idx_to_str[indice]

    def __len__(self):
        return len(self.str_to_idx)


cdef class Variable:

    def __init__(self, str name):
        self.name = name
        self.type = None
        self.values = None
        self.category_indexer = CategoryIndexer()

    def set_values(self, int max_iter, var_type_or_categories):
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
