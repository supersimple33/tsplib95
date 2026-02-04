# -*- coding: utf-8 -*-
import pytest

from tsplib95 import matrix
import numpy as np


def test_base_matrix_requires_get_index_implmentation():
    with pytest.raises(TypeError):
        m = matrix.Matrix(range(1, 10), 3)


@pytest.mark.parametrize('i,j', [
    (99, 1),
    (99, 99),
    (1, 99),
    (-99, 1),
    (-99, -99),
    (1, -99),
])
def test_matrix_value_at_out_of_bounds(i, j):
    m = matrix.FullMatrix(range(1, 10), 3)
    with pytest.raises(IndexError):
        assert m.value_at(i, j)


# 1 2 3
# 4 5 6
# 7 8 9
@pytest.mark.parametrize('i,j,v', [
    (0, 0, 1),
    (0, 2, 3),
    (1, 1, 5),
    (1, 2, 6),
    (2, 0, 7),
    (2, 2, 9),
])
def test_full_matrix(i, j, v):
    m = matrix.FullMatrix(range(1, 10), 3)
    assert m[i, j] == v


def test_full_matrix_to_numpy():
    m = matrix.FullMatrix(range(1, 10), 3)
    np_array = m.to_numpy()
    assert np.all(np_array == np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


# 1 2 3
#   4 5
#     6
@pytest.mark.parametrize('i,j,v', [
    (0, 0, 1),
    (0, 2, 3),
    (1, 1, 4),
    (1, 2, 5),
    (2, 0, 3),
    (2, 2, 6),
])
def test_upper_diag_row(i, j, v):
    m = matrix.UpperDiagRow(range(1, 7), 3)
    assert m[i, j] == v


def test_upper_diag_row_to_numpy():
    m = matrix.UpperDiagRow(range(1, 7), 3)
    np_array = m.to_numpy()
    assert np.all(np_array == np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]]))


# 1
# 2 3
# 4 5 6
@pytest.mark.parametrize('i,j,v', [
    (0, 0, 1),
    (0, 2, 4),
    (1, 1, 3),
    (1, 2, 5),
    (2, 0, 4),
    (2, 2, 6),
])
def test_lower_diag_row(i, j, v):
    m = matrix.LowerDiagRow(range(1, 7), 3)
    assert m[i, j] == v


def test_lower_diag_row_to_numpy():
    m = matrix.LowerDiagRow(range(1, 7), 3)
    np_array = m.to_numpy()
    assert np.all(np_array == np.array([[1, 2, 4], [2, 3, 5], [4, 5, 6]]))


# _ 1 2 3
#   _ 4 5
#     _ 6
#       _
@pytest.mark.parametrize('i,j,v', [
    (0, 0, 0),
    (0, 3, 3),
    (1, 1, 0),
    (1, 2, 4),
    (1, 3, 5),
    (3, 0, 3),
    (2, 3, 6),
    (3, 3, 0),
])
def test_upper_row(i, j, v):
    m = matrix.UpperRow(range(1, 7), 4)
    assert m[i, j] == v


def test_upper_row_to_numpy():
    m = matrix.UpperRow(range(1, 7), 4)
    np_array = m.to_numpy()
    assert np.all(
        np_array == np.array([[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]])
    )


# _
# 1 _
# 2 3 _
# 4 5 6 _
@pytest.mark.parametrize('i,j,v', [
    (0, 0, 0),
    (0, 3, 4),
    (1, 1, 0),
    (1, 2, 3),
    (1, 3, 5),
    (3, 0, 4),
    (2, 3, 6),
    (3, 3, 0),
])
def test_lower_row(i, j, v):
    m = matrix.LowerRow(range(1, 7), 4)
    assert m[i, j] == v


def test_lower_row_to_numpy():
    m = matrix.LowerRow(range(1, 7), 4)
    np_array = m.to_numpy()
    assert np.all(
        np_array == np.array([[0, 1, 2, 4], [1, 0, 3, 5], [2, 3, 0, 6], [4, 5, 6, 0]])
    )


# _ 1 2 4
#   _ 3 5
#     _ 6
#       _
@pytest.mark.parametrize('i,j,v', [
    (0, 0, 0),
    (0, 3, 4),
    (1, 1, 0),
    (1, 2, 3),
    (1, 3, 5),
    (3, 0, 4),
    (2, 3, 6),
    (3, 3, 0),
])
def test_upper_col(i, j, v):
    m = matrix.UpperCol(range(1, 7), 4)
    assert m[i, j] == v


def test_upper_col_to_numpy():
    m = matrix.UpperCol(range(1, 7), 4)
    np_array = m.to_numpy()
    assert np.all(
        np_array == np.array([[0, 1, 2, 4], [1, 0, 3, 5], [2, 3, 0, 6], [4, 5, 6, 0]])
    )


# _
# 1 _
# 2 4 _
# 3 5 6 _
@pytest.mark.parametrize('i,j,v', [
    (0, 0, 0),
    (0, 3, 3),
    (1, 1, 0),
    (1, 2, 4),
    (1, 3, 5),
    (3, 0, 3),
    (2, 3, 6),
    (3, 3, 0),
])
def test_lower_col(i, j, v):
    m = matrix.LowerCol(range(1, 7), 4)
    assert m[i, j] == v


def test_lower_col_to_numpy():
    m = matrix.LowerCol(range(1, 7), 4)
    np_array = m.to_numpy()
    assert np.all(
        np_array == np.array([[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]])
    )


# 1 2 4
#   3 5
#     6
@pytest.mark.parametrize('i,j,v', [
    (0, 0, 1),
    (0, 2, 4),
    (1, 1, 3),
    (1, 2, 5),
    (2, 0, 4),
    (2, 2, 6),
])
def test_upper_diag_col(i, j, v):
    m = matrix.UpperDiagCol(range(1, 7), 3)
    assert m[i, j] == v


def test_upper_diag_col_to_numpy():
    m = matrix.UpperDiagCol(range(1, 7), 3)
    np_array = m.to_numpy()
    assert np.all(np_array == np.array([[1, 2, 4], [2, 3, 5], [4, 5, 6]]))


# 1
# 2 4
# 3 5 6
@pytest.mark.parametrize('i,j,v', [
    (0, 0, 1),
    (0, 2, 3),
    (1, 1, 4),
    (1, 2, 5),
    (2, 0, 3),
    (2, 2, 6),
])
def test_lower_diag_col(i, j, v):
    m = matrix.LowerDiagCol(range(1, 7), 3)
    assert m[i, j] == v


def test_lower_diag_col_to_numpy():
    m = matrix.LowerDiagCol(range(1, 7), 3)
    np_array = m.to_numpy()
    assert np.all(np_array == np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]]))
