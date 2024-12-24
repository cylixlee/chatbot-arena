import queue
import warnings

import numpy as np
import pandas as pd

from src.legacy.abdml import AbdBase
from src.preprocessing import *
from src.settings import load_environment_settings

CONFIG = load_environment_settings("environment-settings.toml")


def load_and_preprocess() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(CONFIG.paths.train)
    test = pd.read_parquet(CONFIG.paths.test)

    computation_pipeline = Sequential(
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
        ComputeResponseLengthDifference(),
        ComputeResponseLengthRatio(),
    )

    vectorizer_queue = queue.Queue()

    preprocess_train = Sequential(
        MapColumnValues("winner", {"model_a": 0, "model_b": 1}),
        DropColumns("model_a", "model_b", "language", "scored"),
        EnforceDType("id", "category"),
        computation_pipeline,
        SequentialOnColumns(
            ["prompt", "response_a", "response_b"],
            PairedVectorizationByTfidf(
                vectorizer_queue,
                fit_transform=True,
                analyzer="char_wb",
                max_features=3000,
            ),
        ),
        DropColumns("prompt", "response_a", "response_b"),
    )

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


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train, test = load_and_preprocess()

    base = AbdBase(
        train_data=train,
        test_data=test,
        target_column="winner",
        gpu=False,
        problem_type="classification",
        metric="accuracy",
        seed=42,
        n_splits=5,
        early_stop=True,
        num_classes=2,
        test_prob=True,
        fold_type="SKF",
        weights=None,
        tf_vec=False,
    )

    params = {
        "n_estimators": 2083,
        "learning_rate": 0.02516607127550297,
        "max_depth": 11,
        "num_leaves": 31,
        "n_jobs": -1,
        "min_child_samples": 42,
        "subsample": 0.8085392166316496,
        "colsample_bytree": 0.6281848449949525,
        "lambda_l1": 4.02155452669029,
        "lambda_l2": 0.14096175149815865,
        "min_gain_to_split": 0.2960660809801552,
    }

    mean_oof_label, mean_test_label, *_ = base.Train_ML(params, "LGBM", e_stop=40)
    sample = pd.read_csv(CONFIG.paths.sample)
    sample["winner"] = np.round(mean_test_label).astype("int")
    sample["winner"] = sample["winner"].map({0: "model_a", 1: "model_b"})

    sample.to_csv("submission_refactored.csv", index=False)


# Guideline recommended Main Guard
if __name__ == "__main__":
    main()
