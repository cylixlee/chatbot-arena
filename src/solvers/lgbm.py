from dataclasses import dataclass
from pprint import pprint

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.solvers import ProblemSolver, ProblemSolution

__all__ = ["LGBMParams", "LGBMSolver"]


@dataclass
class LGBMParams(object):
    n_estimators: int
    learning_rate: float
    max_depth: int
    num_leaves: int
    min_child_samples: int
    subsample: float
    column_sample_by_tree: float
    lambda_l1: float
    lambda_l2: float
    min_gain_to_split: float
    early_stopping: int

    n_splits: int = 5
    """ Number of folds, used in Stratified K-Fold cross validator. """

    random_state: int | None = None
    """ Random state used in Stratified K-Fold cross validator. Set to an int for reproducible results. """


class LGBMSolver(ProblemSolver[LGBMParams]):
    _x_train: pd.DataFrame
    _y_train: pd.DataFrame
    _x_test: pd.DataFrame

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, target_column: str) -> None:
        self._x_train = train.drop(target_column, axis=1)
        self._x_test = train[target_column]
        self._x_test = test

    def solve(self, params: LGBMParams) -> ProblemSolution:
        print("Starting LGBMSolver with parameters:")
        pprint(params)

        cross_validator = StratifiedKFold(params.n_splits, shuffle=True, random_state=params.random_state)
