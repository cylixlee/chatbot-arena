import queue
import warnings

import numpy as np
import pandas as pd

from src.legacy.abdml import AbdBase
from src.preprocessing import *
from src.settings import load_environment_settings

CONFIG = load_environment_settings("environment-settings.toml")


def load_and_preprocess() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the data from binaries (i.e. parquet files) and perform preprocessing operations on them.

    Returns: a DataFrame containing the train data and the other containing test.
    """
    train = pd.read_parquet(CONFIG.paths.train)
    test = pd.read_parquet(CONFIG.paths.test)

    # Computation pipelines.
    #
    # There are 3 text column in the data: "prompt", "response_a" and "response_b". The computation part is to
    # compute the features of those text columns manually (e.g. word count, average word length, etc.) and create new
    # columns to store them.
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
        # Former defined computation pipelines are applied here. We define them separately because they can be reused
        # in the preprocess_text pipeline.
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

    # Similar to process_train so we don't comment this one in detail.
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
    with warnings.catch_warnings():  # warnings in preprocessing stage are ignored.
        warnings.simplefilter("ignore")
        train, test = load_and_preprocess()

    # Calling the encapsulated AbdBase model, same in the legacy code.

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
