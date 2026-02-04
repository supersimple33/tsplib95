# -*- coding: utf-8 -*-
from . import utils
from abc import ABC, abstractmethod
import numpy as np


class Matrix(ABC):
    """A square matrix created from a list of numbers.

    Elements are accessible using matrix notation. Negative indexing is not
    allowed.

    :param list numbers: the elements of the matrix
    :param int size: the width (also height) of the matrix
    :param int min_index: the minimum index
    """

    def __init__(self, numbers, size, min_index=0):
        self.numbers = list(numbers)
        self.size = size
        self.min_index = min_index

    def __getitem__(self, key):
        return self.value_at(*key)

    def value_at(self, i, j):
        """Get the element at row *i* and column *j*.

        :param int i: row
        :param int j: column
        :return: value of element at (i,j)
        """
        i -= self.min_index
        j -= self.min_index
        if not self.is_valid_row_column(i, j):
            raise IndexError(f"({i}, {j}) is out of bounds")
        index = self.get_index(i, j)
        return self.numbers[index]

    def is_valid_row_column(self, i, j):
        """Return True if (i,j) is a row and column within the matrix.

        :param int i: row
        :param int j: column
        :return: whether (i,j) is within the bounds of the matrix
        :rtype: bool
        """
        return 0 <= i < self.size and 0 <= j < self.size

    @abstractmethod
    def get_index(self, i, j):
        """Return the linear index for the element at (i,j).

        :param int i: row
        :param int j: column
        :return: linear index for element (i,j)
        :rtype: int
        """
        raise NotImplementedError()

    @abstractmethod
    def to_numpy(self, dtype=np.float64):
        """Returns a 2D NumPy array containing the elements of the matrix."""
        raise NotImplementedError()


class FullMatrix(Matrix):
    """A complete square matrix.

    :param list numbers: the elements of the matrix
    :param int size: the width (also height) of the matrix
    :param int min_index: the minimum index
    """

    def get_index(self, i, j):
        return i * self.size + j

    def to_numpy(self, dtype=np.float64):
        data = np.asarray(self.numbers, dtype=dtype)
        return data.reshape((self.size, self.size))


class HalfMatrix(Matrix):
    """A triangular half-matrix.

    :param list numbers: the elements of the matrix
    :param int size: the width (also height) of the matrix
    :param int min_index: the minimum index
    """

    #: True if the half-matrix includes the diagonal
    has_diagonal = True

    def value_at(self, i, j):
        if i == j and not self.has_diagonal:
            return 0
        i, j = self._fix_indices(i, j)
        return super().value_at(i, j)

    @abstractmethod
    def _fix_indices(self, i, j):
        raise NotImplementedError()


class UpperDiagRow(HalfMatrix):
    """Upper-triangular matrix that includes the diagonal.

    :param list numbers: the elements of the matrix
    :param int size: the width (also height) of the matrix
    :param int min_index: the minimum index
    """

    has_diagonal = True

    def _fix_indices(self, i, j):
        i, j = (j, i) if i > j else (i, j)
        if not self.has_diagonal:
            j -= 1
        return i, j

    def get_index(self, i, j):
        n = self.size - int(not self.has_diagonal)
        return utils.integer_sum(n, n - i) + (j - i)

    def to_numpy(self, dtype=np.float64):
        n = self.size
        data = np.asarray(self.numbers, dtype=dtype)

        k = 0 if self.has_diagonal else 1
        iu, ju = np.triu_indices(n, k=k)

        offset = iu * (n - k) - (iu * (iu - 1)) // 2
        within = (ju - iu) - k
        lin = offset + within

        out = np.empty((n, n), dtype=dtype)
        if not self.has_diagonal:
            np.fill_diagonal(out, 0)

        out[iu, ju] = data[lin]
        out[ju, iu] = out[iu, ju]  # mirror
        return out


class LowerDiagRow(HalfMatrix):
    """Lower-triangular matrix that includes the diagonal.

    :param list numbers: the elements of the matrix
    :param int size: the width (also height) of the matrix
    :param int min_index: the minimum index
    """

    has_diagonal = True

    def _fix_indices(self, i, j):
        i, j = (j, i) if i < j else (i, j)
        if not self.has_diagonal:
            i -= 1
        return i, j

    def get_index(self, i, j):
        return utils.integer_sum(i) + j

    def to_numpy(self, dtype=np.float64):
        n = self.size
        data = np.asarray(self.numbers, dtype=dtype)

        k = 0 if self.has_diagonal else 1
        il, jl = np.tril_indices(n, k=k)

        out = np.empty((n, n), dtype=dtype)
        if self.has_diagonal:
            offset = il * (il + 1) // 2
            lin = offset + jl
        else:
            offset = il * (il - 1) // 2
            lin = offset + jl - 1
            np.fill_diagonal(out, 0)

        out[il, jl] = data[lin]
        out[jl, il] = out[il, jl]  # mirror
        return out


class UpperRow(UpperDiagRow):
    """Upper-triangular matrix that does not include the diagonal.

    :param list numbers: the elements of the matrix
    :param int size: the width (also height) of the matrix
    :param int min_index: the minimum index
    """

    has_diagonal = False


class LowerRow(LowerDiagRow):
    """Lower-triangular matrix that does not include the diagonal.

    :param list numbers: the elements of the matrix
    :param int size: the width (also height) of the matrix
    :param int min_index: the minimum index
    """

    has_diagonal = False


class UpperCol(LowerRow):
    pass


class LowerCol(UpperRow):
    pass


class UpperDiagCol(LowerDiagRow):
    pass


class LowerDiagCol(UpperDiagRow):
    pass


TYPES = {
    "FULL_MATRIX": FullMatrix,
    "UPPER_DIAG_ROW": UpperDiagRow,
    "UPPER_ROW": UpperRow,
    "LOWER_DIAG_ROW": LowerDiagRow,
    "LOWER_ROW": LowerRow,
    "UPPER_DIAG_COL": UpperDiagCol,
    "UPPER_COL": UpperCol,
    "LOWER_DIAG_COL": LowerDiagCol,
    "LOWER_COL": LowerCol,
}
