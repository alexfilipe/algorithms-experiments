from __future__ import annotations
from typing import Any

class Matrix:
  """Class to represent a matrix, supporting elementary unary and binary
  operations.

  Args:
    array: A list of lists representing the elements
  """

  def __init__(self, array: list[list[Any]], fillna: Any | None = 0):
    self.array = array
    if not array or not any(row for row in array):
      self.array = []

    for row in self.array:
      if not isinstance(row, list):
        raise ValueError("All rows must be lists")

    rows = len(self.array)
    cols = max(len(row) for row in self.array) if rows else 0
    self.dim = rows, cols

    # Normalize matrix, fill missing entries with zeros
    for entry in self.array:
      if len(entry) != cols:
        entry[:] = entry + [fillna] * (cols - len(entry))

    if rows == 0 or cols == 0:
      self.type = None
    else:
      self.type = type(self.array[0][0])

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    r = "\n".join(" ".join(str(c) for c in r) for r in self.array)
    r += f"\n<linalg.Matrix dim={self.dim} type={self.type}>"
    return r

  def transpose(self) -> Matrix:
    """Returns the transpose of this matrix."""
    rows, cols = self.dim
    return Matrix([[self.array[i][j] for i in range(rows)]
                   for j in range(cols)])

  @property
  def T(self) -> Matrix:
    """Transpose of this matrix."""
    return self.transpose()

  def conj_transpose(self) -> Matrix:
    """Implemented only for complex matrices. Returns the conjugate transpose of
    this matrix."""
    if self.type != complex:
      raise ValueError("Conjugate transpose is only implemented for complex "
                       "matrices")
    raise NotImplementedError

  def __add__(self, m: Matrix) -> Matrix:
    if self.dim != m.dim:
      raise ValueError("Matrices must have the same dimension")
    rows, cols = self.dim
    for i in range(rows):
      for j in range(cols):
        self.array[i][j] += m.array[i][j]
    return self

  def __mul__(self, m: Matrix) -> Matrix:
    raise NotImplementedError("Matrix multiplication not implemented.")

  def determinant(self):
    """Returns the determinant of this matrix."""
    raise NotImplementedError

  def det(self):
    """Returns the determinant of this matrix. Alias for Matrix.determinant"""
    return self.determinant()

  def inverse(self):
    """Returns the inverse of this matrix."""
    raise NotImplementedError
