"""Implementation of efficient and inefficient linear algebra algorithms"""

from __future__ import annotations
import operator as op
from functools import reduce
from numbers import Number
from typing import Any, Type

NUMERICAL_TYPES = {int, float, complex}
ZEROS = {int: 0, float: 0., complex: 0j}
ONES = {int: 1, float: 1., complex: 1+0j}

def dot_product(a: list[Number], b: list[Number]) -> Number:
  """Returns the dot product of the two vectors."""
  if len(a) != len(b):
    raise ValueError("Vectors must have same dimension")
  return sum(x * y for x, y in zip(a, b))

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

  def __init__(self, array: list[list], fillna: Any | None = None):
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
      self.dtype = None
    else:
      # TODO: check for `first` not-None type instead
      self.dtype = type(self.array[0][0])

    if fillna is not None and not isinstance(fillna, self.dtype):
      raise TypeError("fillna type must match matrix type")

    if fillna is None and self.dtype is not None:
      if self.dtype in ZEROS:
        fillna = ZEROS[self.dtype]

    # Normalize matrix, fill missing entries with zeros
    for row in self.array:
      if len(row) != cols:
        if fillna is None and self.dtype is not None:
          raise TypeError("fillna must be provided for nonbasic types (int, "
                          "float, complex, str, None) when there are "
                          "incomplete rows")
        row[:] = row + [fillna] * (cols - len(row))

    # Check if matrix is a single type
    for row in self.array:
      for elt in row:
        if elt is not None and not isinstance(elt, self.dtype):
          raise ValueError("All elements must have the same type")

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    r = "\n".join(" ".join(str(c) for c in r) for r in self.array)
    r += f"\n<linalg.Matrix dim={self.dim} type={self.dtype}>"
    return r

  def __eq__(self, m: Matrix) -> bool:
    """Matrix equality."""
    if not isinstance(m, Matrix):
      raise TypeError("Equality only implemented between matrices")
    if self.dim != m.dim:
      return False
    rows, cols = self.dim
    for i in range(rows):
      for j in range(cols):
        if self.array[i][j] != m.array[i][j]:
          return False
    return True

  def copy(self) -> Matrix:
    """Returns a (shallow) copy of this matrix."""
    rows, cols = self.dim
    return Matrix([[self.array[i][j] for j in range(cols)]
                   for i in range(rows)])

  @classmethod
  def identity(cls, dim: int = 0, dtype: Type = int) -> Matrix:
    """Returns the identity matrix of a given dimension and type."""
    one = ONES.get(dtype, ONES[int])
    zero = ZEROS.get(dtype, ZEROS[int])
    return cls([[one if i == j else zero for j in range(dim)] for i in range(dim)])

  @classmethod
  def id(cls, dim: int = 0) -> Matrix:
    """Alias of Matrix.identity."""
    return cls.identity(dim)

  @classmethod
  def zero(cls, dim: int = 0, dtype: Type = int) -> Matrix:
    """Returns the zero matrix of a given dimension and type."""
    zero = ZEROS.get(dtype, ZEROS[int])
    return cls([[zero for _ in range(dim)] for _ in range(dim)])

  def is_numerical(self) -> bool:
    """Returns True if this matrix is a numerical matrix."""
    return self.dtype in NUMERICAL_TYPES

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
    if not self.is_numerical():
      raise TypeError("Conjugate transpose is only implemented for numerical "
                      "matrices")

    if self.dtype in [int, float]:
      return self.copy()
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
    if not self.is_numerical() or not m.is_numerical():
      raise TypeError("Addition only supported between numerical matrices")
    if self.dim != m.dim:
      raise ValueError("Matrices must have the same dimension")
    rows, cols = self.dim
    for i in range(rows):
      for j in range(cols):
        self.array[i][j] += m.array[i][j]
    return self

  def __mul__(self, m: Matrix) -> Matrix:
    """Matrix multiplication (brute force)."""
    if not isinstance(m, Matrix):
      raise TypeError("Right operand must be a matrix")
    if not self.is_numerical() or not m.is_numerical():
      raise TypeError("Matrix multiplication only implemented for numerical "
                      "matrices")
    if self.dim[1] != m.dim[0]:
      raise ValueError("Number of columns in the left matrix must match number "
                       "of rows in right matrix")
    r = [[0 for _ in range(m.dim[1])] for _ in range(self.dim[0])]
    for i in range(self.dim[0]):
      for j in range(m.dim[1]):
        row = self.array[i]
        col = [m.array[k][j] for k in range(m.dim[0])]
        r[i][j] = dot_product(row, col)
    return Matrix(r)

  def __rmul__(self, a: int | float | complex) -> Matrix:
    """Scalar multiplication."""
    if all(not isinstance(a, t) for t in NUMERICAL_TYPES):
      raise TypeError("Left operand must be a matrix or scalar")
    return Matrix([[a * elt for elt in row] for row in self.array])

  def __pow__(self, n: int) -> Matrix:
    """Integer exponentiation."""
    if not isinstance(n, int):
      raise NotImplementedError("Exponentiation only implemented for integer "
                                "powers")
    return reduce(op.mul, (self for _ in range(n)))

  def __neg__(self) -> Matrix:
    """Negative of this matrix."""
    return -1 * self

  def __sub__(self, m: Matrix) -> Matrix:
    """Matrix subtraction."""
    return self + -m

  def determinant(self):
    """Returns the determinant of this matrix."""
    if not self.is_numerical():
      raise TypeError("Determinant only supported for numerical matrices")
    raise NotImplementedError

  def det(self):
    """Returns the determinant of this matrix. Alias for Matrix.determinant"""
    return self.determinant()

  def inverse(self):
    """Returns the inverse of this matrix."""
    raise NotImplementedError
