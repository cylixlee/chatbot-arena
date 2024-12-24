import string
from abc import ABC, abstractmethod
from dataclasses import dataclass

import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing_extensions import override

try:
    from nltk.corpus import stopwords
except LookupError:
    nltk.download("stopwords")
    from nltk.corpus import stopwords


__all__ = [
    "PreprocessPipeline",
    "Sequential",
    "MapColumnValues",
    "DropColumns",
    "EnforceDType",
    "ColumnPreprocessPipeline",
    "SequentialOnColumns",
    "ComputeLength",
    "ComputeWordCount",
    "ComputeCharCount",
    "ComputePunctuationCount",
    "ComputeCapitalizedCount",
    "ComputeSpecialCharCount",
    "ComputeStopwordsCount",
    "ComputeUniqueWordCount",
    "ComputeWordLengthMean",
    "ComputeWordLengthMedian",
    "ComputeWordLengthMax",
    "ComputeWordLengthMin",
    "ComputeSentenceLengthMean",
    "ComputeSentenceLengthMedian",
    "ComputeSentenceLengthMax",
    "ComputeSentenceLengthMin",
    "ComputeAverageWordLength",
    "ComputeLexicalDiversity",
    "ComputeResponseLengthDifference",
    "ComputeResponseLengthRatio",
]

_stopwords = set(stopwords.words("english"))


class PreprocessPipeline(ABC):
    """
    The base class of all preprocess pipelines. It receives a DataFrame and does some transformations,
    and then return the preprocessed data.
    """

    @abstractmethod
    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        pass


class Sequential(PreprocessPipeline):
    """
    Aggregate a pipeline sequence into a single pipeline.
    """

    _pipelines: list[PreprocessPipeline]

    def __init__(self, *pipelines: PreprocessPipeline) -> None:
        self._pipelines = list(pipelines)

    @override
    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        for pipeline in tqdm(self._pipelines, desc="sequential", leave=False):
            frame = pipeline(frame)
        return frame


@dataclass
class MapColumnValues(PreprocessPipeline):
    """
    Map values of a specific column into new values according to a mapping pattern.
    """

    column: str
    mapping: dict

    @override
    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame[self.column] = frame[self.column].map(self.mapping)
        return frame


class DropColumns(PreprocessPipeline):
    """
    Drop some columns.
    """

    _columns: list[str]

    def __init__(self, *columns: str) -> None:
        self._columns = list(columns)

    @override
    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.drop(columns=self._columns, errors="ignore")


@dataclass
class EnforceDType(PreprocessPipeline):
    """
    Enforce value of one column to be of a specific dtype.
    """

    column: str
    dtype: str

    @override
    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame[self.column] = frame[self.column].astype(self.dtype)
        return frame


class ColumnPreprocessPipeline(ABC):
    """
    The base class of preprocessing pipelines against a certain column.
    """

    @abstractmethod
    def __call__(self, frame: pd.DataFrame, column: str) -> pd.DataFrame:
        pass


class SequentialOnColumns(PreprocessPipeline):
    """
    Execute a sequence of ColumnPreprocessPipeline on several specific columns.
    """

    _columns: list[str]
    _pipelines: list[ColumnPreprocessPipeline]

    def __init__(
        self,
        columns: list[str],
        *pipelines: ColumnPreprocessPipeline,
    ) -> None:
        self._columns = columns
        self._pipelines = list(pipelines)

    @override
    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        for column in tqdm(self._columns, desc="on column", leave=False):
            for pipeline in tqdm(self._pipelines, desc="on pipeline", leave=False):
                frame = pipeline(frame, column)
        return frame


def _column_pipeline_of(f):
    new_column_suffix = f.__name__

    @override
    class Pipeline(ColumnPreprocessPipeline):
        def __call__(self, frame: pd.DataFrame, column: str) -> pd.DataFrame:
            frame[f"{column}{new_column_suffix}"] = frame[column].apply(f)
            return frame

    return Pipeline


def _length(x: str) -> int:
    return len(x)


def _word_count(x: str) -> int:
    return len(x.split())


def _char_count(x: str) -> int:
    return sum(len(word) for word in x.split())


def _punctuation_count(x: str) -> int:
    count = 0
    for char in x:
        if char in string.punctuation:
            count += 1
    return count


def _capitalized_count(x: str) -> int:
    count = 0
    for word in x.split():
        if word.isupper():
            count += 1
    return count


def _special_char_count(x: str) -> int:
    count = 0
    for char in x:
        if not char.isalnum() and not char.isspace():
            count += 1
    return count


def _stopwords_count(x: str) -> int:
    count = 0
    for word in x.split():
        if word.lower() in _stopwords:
            count += 1
    return count


def _unique_word_count(x: str) -> int:
    return len(set(x.split()))


def _word_length_mean(x: str) -> np.floating:
    return np.mean([len(word) for word in x.split()])


def _word_length_median(x: str) -> np.floating:
    return np.median([len(word) for word in x.split()])


def _word_length_max(x: str) -> int:
    return max([len(word) for word in x.split()], default=0)


def _word_length_min(x: str) -> int:
    return min([len(word) for word in x.split()], default=0)


def _sentence_length_mean(x: str) -> np.floating:
    lengths = []
    for sentence in x.split("."):
        sentence = sentence.strip()
        if sentence:
            lengths.append(len(sentence.split()))
    return np.mean(lengths)


def _sentence_length_median(x: str) -> np.floating:
    lengths = []
    for sentence in x.split("."):
        sentence = sentence.strip()
        if sentence:
            lengths.append(len(sentence.split()))
    return np.median(lengths)


def _sentence_length_max(x: str) -> int:
    lengths = []
    for sentence in x.split("."):
        sentence = sentence.strip()
        if sentence:
            lengths.append(len(sentence.split()))
    return max(lengths, default=0)


def _sentence_length_min(x: str) -> int:
    lengths = []
    for sentence in x.split("."):
        sentence = sentence.strip()
        if sentence:
            lengths.append(len(sentence.split()))
    return min(lengths, default=0)


ComputeLength = _column_pipeline_of(_length)
ComputeWordCount = _column_pipeline_of(_word_count)
ComputeCharCount = _column_pipeline_of(_char_count)
ComputePunctuationCount = _column_pipeline_of(_punctuation_count)
ComputeCapitalizedCount = _column_pipeline_of(_capitalized_count)
ComputeSpecialCharCount = _column_pipeline_of(_special_char_count)
ComputeStopwordsCount = _column_pipeline_of(_stopwords_count)
ComputeUniqueWordCount = _column_pipeline_of(_unique_word_count)
ComputeWordLengthMean = _column_pipeline_of(_word_length_mean)
ComputeWordLengthMedian = _column_pipeline_of(_word_length_median)
ComputeWordLengthMax = _column_pipeline_of(_word_length_max)
ComputeWordLengthMin = _column_pipeline_of(_word_length_min)
ComputeSentenceLengthMean = _column_pipeline_of(_sentence_length_mean)
ComputeSentenceLengthMedian = _column_pipeline_of(_sentence_length_median)
ComputeSentenceLengthMax = _column_pipeline_of(_sentence_length_max)
ComputeSentenceLengthMin = _column_pipeline_of(_sentence_length_min)


def _column_pipeline_divide(new: str, numerator: str, denominator: str):
    class Pipeline(ColumnPreprocessPipeline):
        @override
        def __call__(self, frame: pd.DataFrame, column: str) -> pd.DataFrame:
            frame[f"{column}_{new}"] = frame[f"{column}_{numerator}"] / frame[f"{column}_{denominator}"]
            return frame

    return Pipeline


ComputeAverageWordLength = _column_pipeline_divide("avg_word_length", "char_count", "word_count")
ComputeLexicalDiversity = _column_pipeline_divide("lexical_diversity", "unique_word_count", "word_count")


class ComputeResponseLengthDifference(PreprocessPipeline):
    @override
    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame["response_length_diff_a_b"] = frame["response_a_length"] - frame["response_b_length"]
        frame["response_length_diff_b_a"] = frame["response_b_length"] - frame["response_a_length"]
        return frame


class ComputeResponseLengthRatio(PreprocessPipeline):
    @override
    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame["response_length_ratio_a_b"] = frame["response_a_length"] / (frame["response_b_length"] + 1e-6)
        frame["response_length_ratio_b_a"] = frame["response_b_length"] / (frame["response_a_length"] + 1e-6)
        return frame
