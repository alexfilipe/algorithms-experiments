from __future__ import annotations
from typing import Any

SCALAR_TYPES = {int, float, complex}

class Matrix:
  """Class to represent a matrix, supporting elementary unary and binary
  operations.

  Matrices are strictly typed. The type is automatically inferred from the first
  element. This constructor supports incomplete rows and fills them accordingly
  with the parameter `fillna` or zero-like values for basic Python types.

  Args:
    array: A list of lists representing the elements
    fillna: Value to fill for missing elements at the end of each row. Must
    match matrix type.
  """

  def __init__(self, array: list[list[Any]], fillna: Any | None = None):
    self.array = array
    if not array or not any(row for row in array):
      self.array = []

    for row in self.array:
      if not isinstance(row, list):
        raise TypeError("All rows must be lists")

    rows = len(self.array)
    cols = max(len(row) for row in self.array) if rows else 0
    self.dim = rows, cols

    if rows == 0 or cols == 0:
      self.type = None
    else:
      # TODO: check for `first` not-None type instead
      self.type = type(self.array[0][0])

    if fillna is not None and not isinstance(fillna, self.type):
      raise TypeError("fillna type must match matrix type")

    if fillna is None and self.type is not None:
      if self.type is int:
        fillna = 0
      elif self.type is float:
        fillna = 0.
      elif self.type is complex:
        fillna = 0j
      elif self.type is str:
        fillna = ''

    # Normalize matrix, fill missing entries with zeros
    for row in self.array:
      if len(row) != cols:
        if fillna is None and self.type is not None:
          raise TypeError("fillna must be provided for nonbasic types (int, float, complex, str, "
                          "None) when there are incomplete rows")
        row[:] = row + [fillna] * (cols - len(row))

    # Check if matrix is a single type
    for row in self.array:
      for elt in row:
        if elt is not None and not isinstance(elt, self.type):
          raise ValueError("All elements must have the same type")

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
    """Returns the conjugate transpose of this matrix. Implemented only for
    complex matrices."""
    if self.type != complex:
      raise TypeError("Conjugate transpose is only implemented for complex "
                      "matrices")

    transposed = self.transpose().array
    return Matrix([[elt.conjugate() for elt in row] for row in transposed])

  @property
  def H(self) -> Matrix:
    """Conjugate transpose of this matrix."""
    return self.conj_transpose()

  def __add__(self, m: Matrix) -> Matrix:
    """Matrix addition."""
    if not isinstance(m, Matrix):
      raise TypeError("Addition only supported between matrices")
    if self.type not in SCALAR_TYPES and m.type not in SCALAR_TYPES:
      raise TypeError("Addition only supported between numerical type matrices")
    if self.dim != m.dim:
      raise ValueError("Matrices must have the same dimension")

    rows, cols = self.dim
    for i in range(rows):
      for j in range(cols):
        self.array[i][j] += m.array[i][j]
    return self

  def __mul__(self, m: Matrix) -> Matrix:
    """Matrix multiplication."""
    if self.type is not m.type:
      raise TypeError("Matrices must have same type")

    raise NotImplementedError

  def __rmul__(self, a: int | float | complex) -> Matrix:
    """Scalar multiplication."""
    if all(not isinstance(a, t) for t in SCALAR_TYPES):
      raise ValueError("Left operand must be a matrix or scalar")

    return Matrix([[a * elt for elt in row] for row in self.array])

  def determinant(self):
    """Returns the determinant of this matrix."""
    raise NotImplementedError

  def det(self):
    """Returns the determinant of this matrix. Alias for Matrix.determinant"""
    return self.determinant()

  def inverse(self):
    """Returns the inverse of this matrix."""
    raise NotImplementedError
