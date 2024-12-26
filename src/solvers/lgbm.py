from dataclasses import dataclass, asdict

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
    """
    A ProblemSolver using LightGBM Classifier from Microsoft.

    Note this solver does NOT support GPU, due to upstream internal reasons.
    """

    _x_train: pd.DataFrame
    _y_train: pd.DataFrame
    _x_test: pd.DataFrame

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, target_column: str) -> None:
        self._x_train = train.drop(target_column, axis=1)
        self._y_train = train[target_column]
        self._x_test = test

    def solve(self, params: LGBMParams) -> ProblemSolution:
        train_scores = []  # The accuracies on the train set of each fold.
        oof_scores = []  # The accuracies on the validation set of each fold

        # The probabilities of model b winning the competition. These probabilities are made by models trained each
        # fold, the mean value of which will be the basis of the final predictions.
        model_b_confidence = np.zeros((len(self._x_test), params.n_splits))

        # Cross validation.
        #
        # Because the amount of data provided is not that huge, cross validation is applied to observe the model
        # performance more precisely. Stratified K-fold validator is used to split the train data into folds with
        # preserved percentage of samples for each class.
        cross_validator = StratifiedKFold(params.n_splits, shuffle=True, random_state=params.random_state)
        with tqdm(total=params.n_splits, desc="Stratified K Fold") as progress:  # adds a progress bar
            # On each fold, we train a model from the ground up, and collect statistics about the metrics.
            for fold, (train_index, validate_index) in enumerate(cross_validator.split(self._x_train, self._y_train)):
                x_train, x_validate = self._x_train.iloc[train_index], self._x_train.iloc[validate_index]
                y_train, y_validate = self._y_train.iloc[train_index], self._y_train.iloc[validate_index]

                # The model parameters.
                #
                # LGBMParams contains not only the parameters needed by the underlying LGBMClassifier, but also some
                # additional ones to control the behavior of solve(). We need to delete them before passing the
                # params onto the model.
                #
                # noinspection PyTypeChecker
                model_params = asdict(params)
                del model_params["n_splits"]
                del model_params["early_stop"]
                # random_state is not del-ed here because there's also a parameter in LGBMClassifier with the same
                # name, and is for reproducible results, too.

                # Create and train the model.
                #
                # LGBMSolver utilizes the existing LGBMClassifier from "lightbgm" package. Early stopping feature is
                # supported by passing a callback provided by the package. For every params.early_stop rounds,
                # if the validation score doesn't improve by min_delta (in this case, 0.0), the training will be
                # stopped.
                model = LGBMClassifier(**model_params, verbose=-1)
                callbacks = None
                if params.early_stop > 0:
                    callbacks = [early_stopping(stopping_rounds=params.early_stop, verbose=False)]
                model.fit(x_train, y_train, eval_set=[(x_validate, y_validate)], callbacks=callbacks)

                # Test model and collect statistics.
                #
                # Once we've trained the model, we use it to predict on the train set and the validation set. The
                # former is used to calculate the accuracy of the model on the train set, and the latter is collected
                # to make a better OOF (out-of-fold) accuracy.
                predict_train = model.predict(x_train)
                predict_validate = model.predict(x_validate)
                train_scores.append(accuracy_score(y_train, (predict_train > 0.5).astype(int)))
                oof_scores.append(accuracy_score(y_validate, (predict_validate > 0.5).astype(int)))

                # Make predictions.
                #
                # Predictions on the test set (which, in the competition, doesn't have a "right answer") are made in
                # probabilities. The classification confidence of the category 1 ("mode_b" the winner) is collected,
                # and the mean value of these probabilities will be the basis of the final prediction.
                model_b_confidence[:, fold] = model.predict_proba(self._x_test)[:, 1]

                # output train accuracy and OOF accuracy in the progress bar.
                progress.set_postfix({"Train": f"{train_scores[-1]:.4f}", "OOF": f"{oof_scores[-1]:.4f}"})
                progress.update()

        # When the training is over, we take mean values of train accuracies and OOF accuracies as the final
        # performance metrics.
        mean_train_scores = f"{np.mean(train_scores):.4f}"
        mean_oof_scores = f"{np.mean(oof_scores):.4f}"
        print(f"Overall Train accuracy {mean_train_scores}")
        print(f"Overall OOF accuracy {mean_oof_scores}")

        final_predictions = model_b_confidence.mean(axis=1).round().astype(int)
        return ProblemSolution(final_predictions)
