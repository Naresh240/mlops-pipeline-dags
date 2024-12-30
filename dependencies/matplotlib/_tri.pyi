# This is a private module implemented in C++
from typing import final

import numpy as np
import numpy.typing as npt

@final
class TrapezoidMapTriFinder:
    def __init__(self, triangulation: Triangulation): ...
    def find_many(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> npt.NDArray[np.int_]: ...
    def get_tree_stats(self) -> list[int | float]: ...
    def initialize(self) -> None: ...
    def print_tree(self) -> None: ...

@final
class TriContourGenerator:
    def __init__(self, triangulation: Triangulation, z: npt.NDArray[np.float64]): ...
    def create_contour(self, level: float) -> tuple[list[float], list[int]]: ...
    def create_filled_contour(self, lower_level: float, upper_level: float) -> tuple[list[float], list[int]]: ...

@final
class Triangulation:
    def __init__(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        triangles: npt.NDArray[np.int_],
        mask: npt.NDArray[np.bool_] | tuple[()],
        edges: npt.NDArray[np.int_] | tuple[()],
        neighbors: npt.NDArray[np.int_] | tuple[()],
        correct_triangle_orientation: bool,
    ): ...
    def calculate_plane_coefficients(self, z: npt.ArrayLike) -> npt.NDArray[np.float64]: ...
    def get_edges(self) -> npt.NDArray[np.int_]: ...
    def get_neighbors(self) -> npt.NDArray[np.int_]: ...
    def set_mask(self, mask: npt.NDArray[np.bool_] | tuple[()]) -> None: ...