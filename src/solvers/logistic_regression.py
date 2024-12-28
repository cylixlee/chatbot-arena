import os
import queue
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from optuna import Trial
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim
from tqdm import tqdm, trange
from typing_extensions import override

from src.preprocessing import *
from src.solvers import ProblemSolution, OptunableProblemSolver

__all__ = ["LogisticRegressor", "LogisticRegressionParams", "LogisticRegressionSolver"]


class LogisticRegressor(nn.Module):
    def __init__(self, in_features: int, middle_features: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, middle_features),
            nn.Linear(middle_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@dataclass
class LogisticRegressionParams(object):
    middle_features: int
    learning_rate: float
    n_splits: int
    epochs: int
    random_state: int


class LogisticRegressionSolver(OptunableProblemSolver[LogisticRegressionParams]):
    # noinspection PyMethodOverriding
    @classmethod
    @override
    def prepare_parameter(cls, trial: Trial, n_splits: int, epochs: int, random_state: int) -> LogisticRegressionParams:
        middle_features = trial.suggest_int("middle_features", low=256, high=4096)
        learning_rate = trial.suggest_float("learning_rate", low=1e-4, high=1e-2)
        return LogisticRegressionParams(middle_features, learning_rate, n_splits, epochs, random_state)

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
            DropNaN(),
            MapColumnValues("winner", {"model_a": 0, "model_b": 1}),
            DropColumns("model_a", "model_b", "language", "scored", "id"),

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
            DropNaN(),
            MinMaxScaleAll(),
        )
        # fmt: on

        # Similar to process_train. We don't comment this one in detail.
        preprocess_test = Sequential(
            DropNaN(),
            DropColumns("model_a", "model_b", "language", "scored", "id"),
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
            DropNaN(),
            MinMaxScaleAll(),
        )

        return preprocess_train(train), preprocess_test(test)

    _x_train: np.ndarray
    _y_train: np.ndarray
    _x_text: torch.Tensor
    _device: torch.device

    def __init__(
        self,
        train: str | os.PathLike,
        test: str | os.PathLike,
        target_column: str,
        gpu: bool | None = None,
    ) -> None:
        train, test = self.__class__.preprocess_cached(train, test)
        self._x_train = train.drop(target_column, axis=1).to_numpy(dtype=np.float32)
        self._y_train = train[target_column].to_numpy(dtype=np.float32)
        if gpu is None:
            gpu = torch.cuda.is_available()
        self._device = torch.device("cuda" if gpu else "cpu")
        self._x_test = torch.from_numpy(test.to_numpy(dtype=np.float32)).to(self._device)

    def solve(self, params: LogisticRegressionParams) -> ProblemSolution:
        train_accuracies = []  # The accuracies on the train set of each fold.
        oof_accuracies = []  # The accuracies on the validation set of each fold
        #
        # # The probabilities of model b winning the competition. These probabilities are made by models trained each
        # # fold, the mean value of which will be the basis of the final predictions.
        # model_b_confidence = np.zeros((len(self._x_test), params.n_splits))
        regression_confidence = []

        cross_validator = StratifiedKFold(params.n_splits, shuffle=True, random_state=params.random_state)
        with tqdm(total=params.n_splits, desc="StratifiedKFold", leave=False) as progress:  # adds a progress bar
            for fold, (train_index, validate_index) in enumerate(cross_validator.split(self._x_train, self._y_train)):
                x_train, x_validate = self._x_train[train_index], self._x_train[validate_index]
                y_train, y_validate = self._y_train[train_index], self._y_train[validate_index]

                x_train = torch.from_numpy(x_train).to(self._device)
                y_train = torch.from_numpy(y_train).to(self._device)
                x_validate = torch.from_numpy(x_validate).to(self._device)
                y_validate = torch.from_numpy(y_validate).to(self._device)

                model = LogisticRegressor(x_train.shape[1], params.middle_features).to(self._device)
                criterion = nn.BCEWithLogitsLoss().to(self._device)
                optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

                model.train()
                for _ in trange(params.epochs, desc="Epoch", leave=False):
                    y_predict = model(x_train).view(-1)
                    loss = criterion(y_predict, y_train)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                predict_train = F.sigmoid(model(x_train).view(-1)).floor()
                correct_train = torch.eq(predict_train, y_train).sum().item()
                train_accuracy = correct_train / len(x_train)
                train_accuracies.append(train_accuracy)

                predict_validate = F.sigmoid(model(x_validate).view(-1)).floor()
                correct_validate = torch.eq(predict_validate, y_validate).sum().item()
                validate_accuracy = correct_validate / len(x_validate)
                oof_accuracies.append(validate_accuracy)

                #
                # # Make predictions.
                # #
                # # Predictions on the test set (which, in the competition, doesn't have a "right answer") are made in
                # # probabilities. The classification confidence of the category 1 ("mode_b" the winner) is collected,
                # # and the mean value of these probabilities will be the basis of the final prediction.
                # model_b_confidence[:, fold] = model.predict_proba(self._x_test)[:, 1]
                #
                # # output train accuracy and OOF accuracy in the progress bar.
                # progress.set_postfix({"Train": f"{train_accuracies[-1]:.4f}", "OOF": f"{oof_accuracies[-1]:.4f}"})
                model.eval()
                confidence = F.sigmoid(model(self._x_test).view(-1))
                regression_confidence.append(confidence)

                progress.set_postfix({"Train": f"{train_accuracy:.4f}", "OOF": f"{validate_accuracy:.4f}"})
                progress.update()

        # When the training is over, we take mean values of train accuracies and OOF accuracies as the final
        # performance metrics.
        mean_train_accuracy = np.mean(train_accuracies)
        mean_oof_accuracy = np.mean(oof_accuracies)
        print(f"Overall Train accuracy {mean_train_accuracy:.4f}")
        print(f"Overall OOF accuracy {mean_oof_accuracy:.4f}")

        final_predictions = torch.stack(regression_confidence).mean(dim=1).round().to(torch.int).cpu().numpy()
        # noinspection PyTypeChecker
        return ProblemSolution(mean_oof_accuracy, final_predictions)
