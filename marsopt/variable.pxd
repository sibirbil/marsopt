cimport numpy as cnp

cdef class CategoryIndexer:
    cdef dict str_to_idx
    cdef dict idx_to_str
    cdef int next_idx

    cpdef cnp.ndarray get_indices(self, list strings)
    cpdef str get_strings(self, int indice)


cdef class Variable:
    cdef public str name
    cdef public object type
    cdef public cnp.ndarray values
    cdef public CategoryIndexer category_indexer
