# -*- coding: utf-8 -*-
import pytest

from tsplib95 import models
from tsplib95 import distances


@pytest.mark.parametrize('start,end,dist,exc', [
    ((3, 4), (3, 4), 0, None),
    ((0, 0), (3, 4), 5, None),
    ((3, 4), (0, 0), 5, None),
    ((-3, -4), (3, 4), 10, None),
    ((0, 1), (0, 1, 2), None, ValueError),
    ((0, 1, 2), (0, 1), None, ValueError),
])
def test_euclidean(start, end, dist, exc):
    if exc:
        with pytest.raises(exc):
            distances.euclidean(start, end)
    else:
        assert distances.euclidean(start, end) == dist


@pytest.mark.parametrize('start,end,dist,exc', [
    ((3, 4), (3, 4), 0, None),
    ((0, 0), (3, 4), 7, None),
    ((3, 4), (0, 0), 7, None),
    ((-3, -4), (3, 4), 14, None),
    ((0, 1), (0, 1, 2), None, ValueError),
    ((0, 1, 2), (0, 1), None, ValueError),
])
def test_manhattan(start, end, dist, exc):
    if exc:
        with pytest.raises(exc):
            distances.manhattan(start, end)
    else:
        assert distances.manhattan(start, end) == dist


@pytest.mark.parametrize('start,end,dist,exc', [
    ((3, 4), (3, 4), 0, None),
    ((0, 0), (3, 4), 4, None),
    ((3, 4), (0, 0), 4, None),
    ((-3, -4), (3, 4), 8, None),
    ((0, 1), (0, 1, 2), None, ValueError),
    ((0, 1, 2), (0, 1), None, ValueError),
])
def test_maximum(start, end, dist, exc):
    if exc:
        with pytest.raises(exc):
            distances.maximum(start, end)
    else:
        assert distances.maximum(start, end) == dist


@pytest.mark.parametrize('start,end,dist,exc', [
    ((3, 4), (3, 4), 1, None),
    ((0, 0), (3, 4), 557, None),
    ((3, 4), (0, 0), 557, None),
    ((-3, -4), (3, 4), 1113, None),
    ((0, 1), (0, 1, 2), None, ValueError),
    ((0, 1, 2), (0, 1), None, ValueError),
])
def test_geographical(start, end, dist, exc):
    if exc:
        with pytest.raises(exc):
            distances.geographical(start, end)
    else:
        assert distances.geographical(start, end) == dist


@pytest.mark.parametrize('pfile,answer', [
    ('data/pcb442.tsp', 221440),
    ('data/gr666.tsp', 423710),
    ('data/att532.tsp', 309636),
])
def test_verifcation_problems(read_problem_text, pfile, answer):
    problem_text = read_problem_text(pfile)
    problem = models.StandardProblem.parse(problem_text)
    assert problem.trace_canonical_tour() == answer
