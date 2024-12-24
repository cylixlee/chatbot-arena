from dataclasses import dataclass, asdict
from pprint import pprint

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.solvers import ProblemSolver, ProblemSolution

__all__ = ["LGBMParams", "LGBMSolver"]


@dataclass
class LGBMParams(object):
    # Parameters required by LGBMClassifier
    n_estimators: int
    learning_rate: float
    max_depth: int
    num_leaves: int
    n_jobs: int
    min_child_samples: int
    subsample: float
    colsample_bytree: float
    lambda_l1: float
    lambda_l2: float
    min_gain_to_split: float

    # Additional parameters used by the LGBMSolver itself.
    early_stop: int
    """ Early stopping rounds, only applied when positive. """

    n_splits: int = 5
    """ Number of folds, used in Stratified K-Fold cross validator. """

    random_state: int | None = None
    """ Random state used in Stratified K-Fold cross validator. Set to an int for reproducible results. """


class LGBMSolver(ProblemSolver[LGBMParams]):
    _x_train: pd.DataFrame
    _y_train: pd.DataFrame
    _x_test: pd.DataFrame
    _gpu: bool

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, target_column: str, gpu: bool) -> None:
        self._x_train = train.drop(target_column, axis=1)
        self._y_train = train[target_column]
        self._x_test = test
        self._gpu = gpu

    def solve(self, params: LGBMParams) -> ProblemSolution:
        print("Starting LGBMSolver with parameters:")
        pprint(params)

        train_scores = []
        oof_scores = []
        all_models = []
        oof_predictions = np.zeros(len(self._y_train))
        test_prediction_proba = np.zeros((len(self._x_test), params.n_splits))
        cross_validator = StratifiedKFold(params.n_splits, shuffle=True, random_state=params.random_state)
        with tqdm(total=params.n_splits, desc="Stratified K Fold") as progress:
            for fold, (train_index, validate_index) in enumerate(cross_validator.split(self._x_train, self._y_train)):
                x_train, x_validate = self._x_train.iloc[train_index], self._x_train.iloc[validate_index]
                y_train, y_validate = self._y_train.iloc[train_index], self._y_train.iloc[validate_index]

                # noinspection PyTypeChecker
                model_params = asdict(params)
                del model_params["n_splits"]
                del model_params["early_stop"]
                # random_state is not del-ed here because there's also a parameter in LGBMClassifier with the same
                # name, and is for reproducible results, too.

                model = LGBMClassifier(**model_params, verbose=-1, device="gpu" if self._gpu else "cpu")
                callbacks = None
                if params.early_stop > 0:
                    callbacks = [early_stopping(stopping_rounds=params.early_stop, verbose=False)]
                model.fit(x_train, y_train, eval_set=[(x_validate, y_validate)], callbacks=callbacks)

                predict_train, predict_validate = model.predict(x_train), model.predict(x_validate)
                oof_predictions[validate_index] = predict_validate
                train_scores.append(accuracy_score(y_train, (predict_train > 0.5).astype(int)))
                oof_scores.append(accuracy_score(y_validate, (predict_validate > 0.5).astype(int)))
                test_prediction_proba[:, fold] = model.predict_proba(self._x_test)[:, 1]

                print(f"Fold {fold}: Train accuracy {train_scores[-1]:.4f}, OOF accuracy {oof_scores[-1]:.4f}")
                all_models.append(model)
                progress.update()

        mean_train_scores = f"{np.mean(train_scores):.4f}"
        mean_oof_scores = f"{np.mean(oof_scores):.4f}"
        print(f"Overall Train accuracy {mean_train_scores}")
        print(f"Overall OOF accuracy {mean_oof_scores}")

        mean_test_prediction_proba = test_prediction_proba.mean(axis=1)
        return ProblemSolution(oof_predictions, mean_test_prediction_proba)
