import os
import queue
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from typing_extensions import override

from src.preprocessing import *
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

    # Additional parameters used by the LGBMSolver.pickle itself.
    early_stop: int
    """ Early stopping rounds, only applied when positive. """

    n_splits: int = 5
    """ Number of folds, used in Stratified K-Fold cross validator. """

    random_state: int | None = None
    """ Random state used in Stratified K-Fold cross validator. Set to an int for reproducible results. """


class LGBMSolver(ProblemSolver[LGBMParams]):
    """
    A ProblemSolver using LightGBM Classifier from Microsoft.

    This is not an OptunableProblemSolver, since this is the baseline model provided by Sheikh Muhammad Abdullah
    (@abdmental01 on Kaggle).

    Note this solver does NOT support GPU, due to upstream internal reasons.
    """

    @classmethod
    @override
    def preprocess_raw(cls, train: str | os.PathLike, test: str | os.PathLike) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = pd.read_parquet(train)
        test = pd.read_parquet(test)

        # Computation pipelines.
        #
        # There are 3 text column in the data: "prompt", "response_a" and "response_b". The computation part is to
        # compute the features of those text columns manually (e.g. word count, average word length, etc.) and create
        # new columns to store them.
        #
        # Those new columns will be passed to the model training as input features.
        computation_pipeline = Sequential(
            # do computations sequentially on those three text columns.
            SequentialOnColumns(
                ["prompt", "response_a", "response_b"],
                ComputeLength(),
                ComputeWordCount(),
                ComputeCharCount(),
                ComputeAverageWordLength(),
                ComputePunctuationCount(),
                ComputeCapitalizedCount(),
                ComputeSpecialCharCount(),
                ComputeStopwordsCount(),
                ComputeUniqueWordCount(),
                ComputeLexicalDiversity(),
                ComputeWordLengthMean(),
                ComputeWordLengthMedian(),
                ComputeWordLengthMax(),
                ComputeWordLengthMin(),
                ComputeSentenceLengthMean(),
                ComputeSentenceLengthMedian(),
                ComputeSentenceLengthMax(),
                ComputeSentenceLengthMin(),
            ),
            # moreover, the difference and ratio between two responses are computed.
            ComputeResponseLengthDifference(),
            ComputeResponseLengthRatio(),
        )

        # the queue storing TfidfVectorizers, for PairedVectorizationByTfidf pipeline.
        vectorizer_queue = queue.Queue()

        # fmt: off
        preprocess_train = Sequential(
            # Transformation.
            #
            # Data transformations are done here, for example, transforming the winners ("model_a", "model_b") into
            # numbers (0 or 1), since this is a classification task.
            MapColumnValues("winner", {"model_a": 0, "model_b": 1}),
            DropColumns("model_a", "model_b", "language", "scored"),
            EnforceDType("id", "category"),

            # Computation.
            #
            # Former defined computation pipelines are applied here. We define them separately because they can be
            # reused in the preprocess_text pipeline.
            computation_pipeline,

            # Vectorization.
            #
            # Besides the computed features, we still need to feed the original texts to models, in some form.
            # Vectorization is the solution: we use vectorizer to transform the text into numbers, and then take them as
            # input features.
            #
            # The inner principle of vectorizer is not discussed here. We just import and call them.
            SequentialOnColumns(
                ["prompt", "response_a", "response_b"],
                PairedVectorizationByTfidf(
                    vectorizer_queue,
                    fit_transform=True,
                    analyzer="char_wb",
                    max_features=3000,
                ),
            ),

            # since we've vectorized the text columns, we no longer need them.
            DropColumns("prompt", "response_a", "response_b"),
        )
        # fmt: on

        # Similar to process_train. We don't comment this one in detail.
        preprocess_test = Sequential(
            DropColumns("model_a", "model_b", "language", "scored"),
            EnforceDType("id", "category"),
            computation_pipeline,
            SequentialOnColumns(
                ["prompt", "response_a", "response_b"],
                PairedVectorizationByTfidf(
                    vectorizer_queue,
                    fit_transform=False,
                    analyzer="char_wb",
                    max_features=3000,
                ),
            ),
            DropColumns("prompt", "response_a", "response_b"),
        )

        return preprocess_train(train), preprocess_test(test)

    _x_train: pd.DataFrame
    _y_train: pd.DataFrame
    _x_test: pd.DataFrame

    def __init__(self, train: str | os.PathLike, test: str | os.PathLike, target_column: str) -> None:
        train, test = self.__class__.preprocess_cached(train, test)
        self._x_train = train.drop(target_column, axis=1)
        self._y_train = train[target_column]
        self._x_test = test

    @override
    def solve(self, params: LGBMParams) -> ProblemSolution:
        train_accuracies = []  # The accuracies on the train set of each fold.
        oof_accuracies = []  # The accuracies on the validation set of each fold

        # The probabilities of model b winning the competition. These probabilities are made by models trained each
        # fold, the mean value of which will be the basis of the final predictions.
        model_b_confidence = np.zeros((len(self._x_test), params.n_splits))

        # Cross validation.
        #
        # Because the amount of data provided is not that huge, cross validation is applied to observe the model
        # performance more precisely. Stratified K-fold validator is used to split the train data into folds with
        # preserved percentage of samples for each class.
        cross_validator = StratifiedKFold(params.n_splits, shuffle=True, random_state=params.random_state)
        with tqdm(total=params.n_splits, desc="StratifiedKFold", leave=False) as progress:  # adds a progress bar
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
                # LGBMSolver.pickle utilizes the existing LGBMClassifier from "LightGBM" package. Early stopping
                # feature is supported by passing a callback provided by the package. For every params.early_stop
                # rounds, if the validation score doesn't improve by min_delta (in this case, 0.0), the training will be
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

                train_accuracy = accuracy_score(y_train, (predict_train > 0.5).astype(int))
                oof_accuracy = accuracy_score(y_validate, (predict_validate > 0.5).astype(int))

                print(f"Fold {fold}, Train {train_accuracy}, OOF {oof_accuracy}")

                train_accuracies.append(train_accuracy)
                oof_accuracies.append(oof_accuracy)

                # Make predictions.
                #
                # Predictions on the test set (which, in the competition, doesn't have a "right answer") are made in
                # probabilities. The classification confidence of the category 1 ("mode_b" the winner) is collected,
                # and the mean value of these probabilities will be the basis of the final prediction.
                model_b_confidence[:, fold] = model.predict_proba(self._x_test)[:, 1]

                # output train accuracy and OOF accuracy in the progress bar.
                progress.set_postfix({"Train": f"{train_accuracies[-1]:.4f}", "OOF": f"{oof_accuracies[-1]:.4f}"})
                progress.update()

        # When the training is over, we take mean values of train accuracies and OOF accuracies as the final
        # performance metrics.
        mean_train_accuracy = np.mean(train_accuracies)
        mean_oof_accuracy = np.mean(oof_accuracies)
        print(f"Overall Train accuracy {mean_train_accuracy:.4f}")
        print(f"Overall OOF accuracy {mean_oof_accuracy:.4f}")

        final_predictions = model_b_confidence.mean(axis=1).round().astype(int)
        # noinspection PyTypeChecker
        return ProblemSolution(mean_oof_accuracy, final_predictions)
