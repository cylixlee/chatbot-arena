import os
import unittest
from dataclasses import dataclass

import pandas as pd
from optuna import Trial
from typing_extensions import override

from src import optuning
from src.solvers import OptunableProblemSolver, ProblemSolution


@dataclass
class FakeParams(object):
    a: float
    b: int


class FakeProblemSolver(OptunableProblemSolver[FakeParams]):
    @classmethod
    @override
    def prepare_parameter(cls, trial: Trial, *args, **kwargs) -> FakeParams:
        assert not args and not kwargs, "FakeProblemSolver does not receive more params"
        a = trial.suggest_float("a", low=-2, high=2)
        b = trial.suggest_int("b", low=0, high=2)
        return FakeParams(a, b)

    @classmethod
    @override
    def preprocess_raw(cls, train: str | os.PathLike, test: str | os.PathLike) -> tuple[pd.DataFrame, pd.DataFrame]:
        # we don't really need this in this test case.
        pass

    @override
    def solve(self, params: FakeParams) -> ProblemSolution:
        # noinspection PyTypeChecker
        solution = ProblemSolution(accuracy=0, predictions=None)
        solution.accuracy = params.a * 2 + params.b
        return solution


class TestOptuning(unittest.TestCase):
    def test_optune_solver(self):
        best_value = optuning.optune_solver(FakeProblemSolver(), 100)
        self.assertAlmostEqual(best_value, 6, places=2)
