import warnings

import pandas as pd

from src.preprocessing import *
from src.settings import load_environment_settings

CONFIG = load_environment_settings("environment-settings.toml")


def load_and_preprocess() -> tuple[pd.DataFrame, pd.DataFrame]:
    warnings.filterwarnings("ignore")
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

    preprocess_test = Sequential(
        DropColumns("model_a", "model_b", "language", "scored"),
        EnforceDType("id", "category"),
        computation_pipeline,
    )

    preprocess_train = Sequential(
        MapColumnValues("winner", {"model_a": 0, "model_b": 1}),
        preprocess_test,
        computation_pipeline,
    )

    warnings.resetwarnings()
    return preprocess_train(train), preprocess_test(test)


def main() -> None:
    train, test = load_and_preprocess()
    train.to_csv("train_refactored.csv", escapechar="|")
    test.to_csv("test_refactored.csv", escapechar="|")


# Guideline recommended Main Guard
if __name__ == "__main__":
    main()
