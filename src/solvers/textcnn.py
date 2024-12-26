import os
from dataclasses import dataclass

import pandas as pd
from optuna import Trial
from typing_extensions import override

from src.solvers import OptunableProblemSolver, ProblemSolution


@dataclass
class TextCNNParams(object):
    pass


class TextCNNSolver(OptunableProblemSolver[TextCNNParams]):
    @classmethod
    @override
    def prepare_parameter(cls, trial: Trial, *args, **kwargs) -> TextCNNParams:
        pass

    @classmethod
    @override
    def preprocess_raw(cls, train: str | os.PathLike, test: str | os.PathLike) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass

    @override
    def solve(self, params: TextCNNParams) -> ProblemSolution:
        pass
