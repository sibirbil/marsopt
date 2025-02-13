import numpy as np
from numpy.typing import NDArray
from typing import List, Union


class CategoryIndexer:
    def __init__(self):
        self.str_to_idx = {}
        self.next_idx = 0

    def get_indices(self, strings):
        result = []
        for s in strings:
            if s not in self.str_to_idx:
                self.str_to_idx[s] = self.next_idx
                self.next_idx += 1
            result.append(self.str_to_idx[s])
        return result

    def get_strings(self, indices):
        idx_to_str = {v: k for k, v in self.str_to_idx.items()}
        return [idx_to_str[i] for i in indices]

    def reset(self):
        self.str_to_idx.clear()
        self.next_idx = 0

    def __len__(self):
        return len(self.str_to_idx)


class Parameter:
    __slots__ = ["name", "type", "values", "category_indexer"]

    def __init__(self, name: str):
        self.name: str = name
        self.type: type = None
        self.values: NDArray = None
        self.category_indexer = CategoryIndexer()

    def __repr__(self) -> str:
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
                self.values = np.empty(shape=(max_iter, max_indice), dtype=np.float64)
                self.type = type(categories)
            elif total_cats < max_indice:
                self.values = np.vstack(
                    (self.values, np.zeros(shape=(max_iter, 1), dtype=np.float64))
                )
        else:
            raise ValueError(
                f"Unsupported parameter type: {type(param_type_or_categories)}"
            )
