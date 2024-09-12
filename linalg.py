"""Implementation of efficient and inefficient linear algebra algorithms"""

from __future__ import annotations
import sys
import itertools
import operator
from functools import reduce
from typing import Iterable


NoneType = type(None)
MatrixType = NoneType | int | float | complex | str
Scalar = int | float | complex
MatrixSlice = int | slice | tuple[int | slice, int | slice]

# Types in order of precedence.
TYPES: list[type[MatrixType]] = [NoneType, int, float, complex, str]
NUMERICAL_TYPES: list[type[MatrixType]] = [int, float, complex]
ZEROS: dict[type[MatrixType], Scalar] = {int: 0, float: 0., complex: 0j}
ONES: dict[type[MatrixType], Scalar] = {int: 1, float: 1., complex: 1+0j}
MULTIPLICATION_ALGORITHM: str = 'naive'  # 'naive' or 'strassen'


def dot_product(a: list[Scalar], b: list[Scalar]) -> Scalar:
  """Returns the dot product of the two vectors."""
  if len(a) != len(b):
    raise ValueError("Vectors must have same dimension")
  return sum(x * y for x, y in zip(a, b))

dot = dot_product


def is_numerical(value: MatrixType) -> bool:
  """Returns True if the value is numerical."""
  return any(isinstance(value, t) for t in NUMERICAL_TYPES)


def type_compatible(value: MatrixType, dtype: type[MatrixType]) -> bool:
  """Returns True if the variables are type compatible."""
  if is_numerical(value) and dtype in NUMERICAL_TYPES:
    # TODO: WARNING! Fix this -- 1.5 is not an integer but this will return True
    return True
  return isinstance(value, dtype)


def is_subset(dtype1: type, dtype2: type, strict: bool = True) -> bool:
  """Returns True if the first numerical type is a strict subset of the second type.

  Args:
    dtype1: First numerical type.
    dtype2: Second numerical type.
    strict: If False, also returns True if the types are equal.
  """
  if not strict and dtype1 == dtype2:
    return True
  return TYPES.index(dtype1) < TYPES.index(dtype2)


class Matrix:
  """Class to represent a matrix, supporting elementary unary and binary operations.

  Matrices are strictly typed. The type is automatically inferred from the first element. Matrix
  types are immutable once they are declared. This constructor supports incomplete rows and fills
  them accordingly with the parameter `fillna` or zero-like values for basic Python types.

  Args:
    data: A (finite) iterable of (finite) iterables of a type supported by this matrix.
    dim: The dimension of the matrix (optional). The data will be truncated or filled if the
      dimension is different than the given data.
    fillna: Value to fill for missing elements at the end of each row. Must match matrix type.
  """

  def __init__(self, data: Iterable[Iterable[MatrixType]], dim: tuple[int, int] | None = None,
               fillna: MatrixType | None = None):
    # TODO data can also be `0` or `1`, and dimension can be explicitly given (add the matrix to the
    # top left corner)
    self.__array: list[list[MatrixType]] = [[]]
    self.dtype: type[MatrixType] = NoneType
    self.dim: tuple[int, int] = (0, 0)

    data = list(data)
    if data and any(row for row in data):
      self.__array = [[] for _ in range(len(data))]

    for i, row in enumerate(data):
      if not isinstance(row, list):
        try:
          self.__array[i] = list(row)
        except TypeError:
          raise TypeError("All rows must be lists or iterables")
      else:
        self.__array[i] = row

    rows = len(self.__array)
    cols = max(len(row) for row in self.__array) if rows else 0
    self.dim = (rows, cols)

    if rows != 0 and cols != 0:
      self.dtype = TYPES[0]
      # Find minimum common type
      for row in self.__array:
        for elt in row:
          if is_subset(self.dtype, type(elt)):
            self.dtype = type(elt)

    if fillna is not None and not type_compatible(fillna, self.dtype):
      raise TypeError(f"fillna type ({type(fillna)}) must be compatible matrix type ({self.dtype})")

    if fillna is None and self.dtype is not NoneType and self.dtype in ZEROS:
      fillna = ZEROS[self.dtype]

    # Normalize matrix, fill missing entries with default value
    for row in self.__array:
      if len(row) != cols:
        if fillna is None and self.dtype is not NoneType:
          raise TypeError("fillna must be provided for nonbasic types (int, float, complex, str, "
                          "NoneType) when there are incomplete rows")
        row[:] = row + [fillna] * (cols - len(row))

    # Check if matrix is a single type
    for row in self.__array:
      for elt in row:
        if elt is not None and not type_compatible(elt, self.dtype):
          raise ValueError("All elements must have the same type")

  def __str__(self):
    rows, cols = self.dim
    colspans = [max(len(str(self.__array[i][j])) for i in range(rows)) for j in range(cols)]
    return "\n".join(" ".join(f"{str(c) : >{colspans[i]}}" for i, c in enumerate(r))
                     for r in self.__array)

  def __repr__(self):
    return (f"{self.__str__()}\n"
            f"<linalg.Matrix dim={self.dim} type={self.dtype} size={self.size()}>")

  def size(self) -> int:
    """Partially deep getsizeof of the contents of this matrix."""
    if self.dim[0] > 0 and self.dim[1] > 0:
      return sum(sys.getsizeof(row) for row in self.__array)
    return 0

  @classmethod
  def from_str(cls, string: str, rowsep: str = '\n', colsep: str = ' ',
               dtype: type[MatrixType] = float) -> Matrix:
    if dtype not in TYPES:
      raise TypeError(f"Type {dtype} not allowed for linalg.Matrix")
    rows = [r for r in string.split(rowsep) if r]
    data = []
    for r in rows:
      typecast = dtype
      if typecast is int:
        typecast = lambda v: int(float(v))
      try:
        data.append([typecast(c) for c in r.split(colsep) if c])
      except Exception as exc:
        raise ValueError(f"Error casting row to type {dtype}: {exc}")
    return cls(data)

  @classmethod
  def from_file(cls, filepath: str, rowsep: str = '\n', colsep: str = ' ',
                dtype: type[MatrixType] = float) -> Matrix:
    with open(filepath, 'r') as f:
      return cls.from_str(rowsep.join(f.readlines()), rowsep, colsep, dtype)

  def __eq__zero(self) -> bool:
    """Equality between matrix and zero matrix."""
    m, n = self.dim
    for i in range(m):
      for j in range(n):
        if self[i,j] != 0:
          return False
    return True

  def __eq__identity(self) -> bool:
    """Equality between matrix and identity matrix."""
    if not self.is_square():
      return False
    n = self.dim[0]
    for i in range(n):
      for j in range(n):
        if i == j and self[i,j] != 1:
          return False
        if i != j and self[i,j] != 0:
          return False
    return True

  def __eq__int(self, M: int) -> bool:
    """Equality between matrix and integer representing matrix."""
    if not self.is_numerical():
      raise TypeError("Equality with integer only implemented for numerical matrices")
    if M == 0:
      return self.__eq__zero()
    if M == 1:
      return self.__eq__identity()
    raise TypeError("Equality only implemented between matrices")

  def __eq__(self, M: Matrix | int) -> bool:
    """Matrix equality.

    Equality is supported between matrices of same dimension, and the integer literals `0` and `1`
    (representing, respectively, the zero and identity matrices of same dimension).
    """
    if not isinstance(M, Matrix):
      if M in [0, 1]:
        return self.__eq__int(M)
      raise TypeError("Equality only implemented between matrices")
    if self.dim != M.dim:
      return False
    rows, cols = self.dim
    for i in range(rows):
      for j in range(cols):
        if self[i,j] != M[i,j]:
          return False
    return True

  def __getitem__(self, index: MatrixSlice) -> MatrixType | list[MatrixType] | Matrix:
    rows, cols = self.dim
    if isinstance(index, slice):
      # Return rows
      start, stop = index.start, index.stop
      return Matrix(self.__array[start:stop])

    if isinstance(index, tuple):
      if len(index) != 2:
        raise IndexError("Index must be an integer or 2-tuple of integers or slices")
      if any(isinstance(i, slice) for i in index):
        raise NotImplementedError("Matrix slicing not implemented")
      i, j = index
      return self.__array[i][j]

    if isinstance(index, int):
      return self.__array[index]

    raise IndexError("Index must be an integer or 2-tuple of integers")

  def __setitem__(
      self,
      index: MatrixSlice,
      value: MatrixType | list[MatrixType] | list[list[MatrixType]] | Matrix,
  ) -> MatrixType | list[MatrixType] | Matrix:

    if isinstance(index, slice):
      raise NotImplementedError("Slice assignment not implemented")

    if isinstance(index, tuple):
      if any(isinstance(i, slice) for i in index):
        raise NotImplementedError("Slice assignment not implemented")
      if len(index) == 2:
        if value is not None and not type_compatible(value, self.dtype):
          raise ValueError(f"Cannot set value of type {type(value)} to matrix of type {self.dtype}")
        # Change main type to accommodate new introduced type
        elif is_numerical(value) and is_subset(self.dtype, type(value)):
          self.dtype = type(value)
        i, j = index
        self.__array[i][j] = value
        return self.__array[i][j]

    if isinstance(index, int):
      if not isinstance(value, list):
        raise ValueError(f"Must set row to list type, not {type(value)}")
      if len(value) != self.dim[1]:
        raise ValueError(f"Must set row to same dimension as matrix (expected {self.dim[1]}, got "
                         f"{len(value)})")
      self.__array[index][:] = value

    raise IndexError("Index must be an integer or 2-tuple of integers")

  def __len__(self) -> int:
    """Returns the number of rows."""
    return self.dim[0]

  def copy(self) -> Matrix:
    """Returns a (shallow) copy of this matrix."""
    rows, cols = self.dim
    return Matrix([[self[i,j] for j in range(cols)] for i in range(rows)])

  @classmethod
  def identity(cls, dim: int | tuple[int, int] = 0, dtype: type[Scalar] = int) -> Matrix:
    """Returns the identity matrix of a given dimension and type.

    An identity matrix is a square matrix containing 1s along its diagonal and 0s elsewhere. For
    compatibility, this method allows square tuples.
    """
    if isinstance(dim, tuple):
      if len(dim) != 2 or dim[0] != dim[1]:
        raise ValueError("first and second dimensions must be equal")
      dim = dim[0]
    one = ONES.get(dtype, ONES[int])
    zero = ZEROS.get(dtype, ZEROS[int])
    return cls([[one if i == j else zero for j in range(dim)] for i in range(dim)])

  id = identity

  @classmethod
  def zero(cls, dim: int | tuple[int, int] = 0, dtype: type[Scalar] = int) -> Matrix:
    """Returns the zero matrix of a given dimension and type.

    A zero matrix is a matrix containing only zeros.
    """
    if isinstance(dim, int):
      dim = (dim, dim)
    zero = ZEROS.get(dtype, ZEROS[int])
    return cls([[zero for _ in range(dim[1])] for _ in range(dim[0])])

  def is_numerical(self) -> bool:
    """Returns True if this matrix is a numerical matrix."""
    return self.dtype in NUMERICAL_TYPES

  def is_square(self) -> bool:
    """Returns True if this matrix is a square matrix."""
    return self.dim[0] == self.dim[1]

  def transpose(self) -> Matrix:
    """Returns the transpose of this matrix."""
    rows, cols = self.dim
    return Matrix([[self[i,j] for i in range(rows)] for j in range(cols)])

  @property
  def T(self) -> Matrix:
    """Transpose of this matrix."""
    return self.transpose()

  def conj_transpose(self) -> Matrix:
    """Returns the conjugate transpose of this matrix. Implemented only for complex matrices."""
    if not self.is_numerical():
      raise TypeError("Conjugate transpose is only implemented for numerical matrices")
    if self.dtype in [int, float]:
      return self.transpose()
    return Matrix([[elt.conjugate() for elt in row] for row in self.transpose().array])

  @property
  def H(self) -> Matrix:
    """Conjugate transpose of this matrix."""
    return self.conj_transpose()

  def __add__(self, M: Matrix | int) -> Matrix:
    """Matrix addition. Supports the integer literals `0` and `1`, representing the zero and
    identity matrices of same dimension, respectively.
    """
    if self.is_square():
      if M == 0:
        return self.copy()
      if M == 1:
        M = Matrix.identity(self.dim, dtype=self.dtype)
    if not isinstance(M, Matrix):
      raise TypeError("Addition only supported between matrices")
    if not self.is_numerical() or not M.is_numerical():
      raise TypeError("Addition only supported between numerical matrices")
    if self.dim != M.dim:
      raise ValueError("Matrices must have same dimension")
    rows, cols = self.dim
    return Matrix([[self[i,j] + M[i,j] for j in range(cols)] for i in range(rows)])

  def strassen_mul(self, M: Matrix) -> Matrix:
    """Multiplication using Strassen's fast matrix multiplication algorithm."""
    raise NotImplementedError("Strassen multiplication not implemented")

  def naive_mul(self, M: Matrix) -> Matrix:
    """Naive matrix multiplication algorithm."""
    prod = Matrix.zero((self.dim[0], M.dim[1]), self.dtype)
    for i in range(self.dim[0]):
      for j in range(M.dim[1]):
        row = self[i]
        col = [M[k,j] for k in range(M.dim[0])]
        prod[i,j] = dot_product(row, col)
    return prod

  def __mul__(self, M: Matrix | Scalar) -> Matrix:
    """Matrix multiplication."""
    if is_numerical(M):
      return self.__rmul__(M)
    if not isinstance(M, Matrix):
      raise TypeError(f"Right operand (type={type(M)}) must be a matrix or scalar")
    if not self.is_numerical() or not M.is_numerical():
      raise TypeError("Matrix multiplication only implemented for numerical matrices")
    if self.dim[1] != M.dim[0]:
      raise ValueError("Scalar of columns in the left matrix must match number of rows in right "
                       "matrix")

    if MULTIPLICATION_ALGORITHM == "strassen":
      return self.strassen_mul(M)
    if MULTIPLICATION_ALGORITHM == "naive":
      return self.naive_mul(M)

    raise NotImplementedError(f"`{MULTIPLICATION_ALGORITHM}` algorithm not implemented")

  def __rmul__(self, a: Scalar) -> Matrix:
    """Scalar multiplication."""
    if a == 0:
      return Matrix.zero(self.dim, dtype=self.dtype)
    if a == 1:
      return self.copy()
    if not is_numerical(a):
      raise TypeError(f"Left operand (type={type(a)}) must be a matrix or scalar")
    return Matrix([[a * elt for elt in row] for row in self.__array])

  def __div(self, a: Scalar, floor: bool = False) -> Matrix:
    """Scalar division implemented for both floor and floating point division."""
    if not is_numerical(a):
      raise ValueError("Division only allowed by scalars")
    if a == 0:
      raise ZeroDivisionError("Division by zero is undefined")
    if a == 1:
      return self.copy()
    if floor:
      return Matrix([[int(elt // a) for elt in row] for row in self.__array])
    return (1 / a) * self

  def __truediv__(self, a: Scalar) -> Matrix:
    """Scalar division."""
    return self.__div(a)

  def __floordiv__(self, a: Scalar) -> Matrix:
    """Scalar floor division."""
    return self.__div(a, floor=True)

  def __neg__(self) -> Matrix:
    """Negative of this matrix."""
    return -1 * self

  def __sub__(self, M: Matrix | int) -> Matrix:
    """Matrix subtraction."""
    if M == 0:
      M = Matrix.zero(self.dim)
    if M == 1:
      M = Matrix.identity(self.dim)
    return self + -M

  def __pow__(self, n: int) -> Matrix:
    """Integer exponentiation."""
    if not isinstance(n, int):
      raise NotImplementedError("Exponentiation only implemented for integer powers")
    if not self.is_square():
      raise ValueError("Exponentiation only defined for square matrices")
    if n == 0:
      return Matrix.identity(self.dim, dtype=self.dtype)
    if n == -1:
      return self.inverse()
    if n < -1:
      return (self ** -1) ** -n
    return reduce(operator.mul, itertools.repeat(self, n))

  def kronecker_prod(self, B: Matrix) -> Matrix:
    """Returns the Kronecker product of the two matrices.

    Given two matrices A (m x n) and B (p x q), the Kronecker product is the block matrix (pm x qn)
    obtained by the following operation:

    A x B = [[A[0,0]*B,  A[0,1]*B,  ...,  A[0,n]*B],
             [A[1,0]*B,  A[1,1]*B,  ...,  A[1,n]*B],
             ...,
             [A[m,0]*B,  A[m,1]*B,  ...,  A[m,n]*B]].
    """
    if not self.is_numerical() or not B.is_numerical():
      raise ValueError("Kronecker product only defined for scalar matrices")
    (m, n), (p, q) = self.dim, B.dim
    prod = [[0 for _ in range(q * n)] for _ in range(p * m)]
    for i in range(m):
      for j in range(n):
        subprod = self[i,j] * B
        for k in range(p):
          for l in range(q):
            prod[k + i*p][l + j*q] = subprod[k,l]
    return Matrix(prod)

  kron = kronecker_prod

  def minor(self, i: int, j: int) -> Matrix:
    """Returns the minor of this matrix at position i,j (matrix obtained by removing the ith row and
    jth column).

    Args:
      i: index of the row to be removed
      j: index of the column to be removed
    """
    # TODO: idea, create MatrixMinor object, which is a subclass of the Matrix class, and only
    # implements the __getitem__ method. This way, we can avoid creating a new matrix object every
    # time we need a minor.
    rows, cols = self.dim
    if i >= rows or j >= cols:
      raise IndexError("Position out of range")
    return Matrix([[self[r,c] for c in range(cols) if c != j] for r in range(rows) if r != i])

  def determinant(self) -> Scalar:
    """Returns the determinant of this matrix."""
    # TODO implement a faster algorithm that doesn't involve creating so many Matrix objects.
    # This algorithm is an oversimplification and has factorial complexity.
    if not self.is_numerical():
      raise TypeError("Determinant only defined for numerical matrices")
    if not self.is_square():
      raise TypeError("Determinant only defined for square matrices")

    dim = self.dim[0]
    if dim == 0:
      return 0
    if dim == 1:
      return self[0,0]
    if dim == 2:
      return self[0,0] * self[1,1] - self[0,1] * self[1,0]

    d = 0
    for i in range(dim):
      r = self[0,i] * self.minor(0, i).determinant()
      if i % 2:
        r *= -1
      d += r
    return d

  det = determinant

  def inverse(self):
    """Returns the inverse of this matrix."""
    if not self.is_numerical():
      raise TypeError("Inverse only defined for numerical matrices")
    if not self.is_square():
      raise TypeError("Inverse only defined for square matrices")
    d = self.determinant()
    if d == 0:
      raise ArithmeticError("Matrix not invertible")
    raise NotImplementedError("Matrix inversion not implemented")

  def rank(self) -> int:
    """Returns the rank of this matrix."""
    raise NotImplementedError("Rank not implemented")


MatrixLike = Matrix | Iterable[MatrixType] | Iterable[Iterable[MatrixType]]


def dim(A: Matrix) -> tuple[int, int]:
  """Returns the dimension of the given matrix."""
  return A.dim


def minor(A: Matrix, i: int, j: int) -> Matrix:
  """Returns the minor of the given matrix at the given position i, j."""
  return A.minor(i, j)


def det(A: Matrix) -> Scalar:
  """Returns the determinant of the given matrix."""
  return A.determinant()


def kronecker_prod(A: Matrix, B: Matrix) -> Matrix:
  """Returns the Kronecker product of the two matrices."""
  return A.kronecker_prod(B)

kron = kronecker_prod
