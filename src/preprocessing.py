"""
The preprocessing module.

This module contains several kinds of preprocessing pipelines. A preprocessing pipeline is an object with a __call__
function receiving a pandas DataFrame, doing some operations on it, and returns it. This design pattern is called the
pipeline pattern or the middleware pattern.

There are two special kinds of pipelines: the Sequential and the SequentialOnColumns. The former takes and wraps
several PreprocessPipelines, and execute them in their definition order when __call__ of the Sequential pipeline is
called.

    Example:

        import pandas as pd
        data = pd.DataFrame(
            [
                ["a", "b", "hello"],
                ["a", "b", "world"],
            ],
            columns=["a", "b", "c"],
        )
        pipeline = Sequential(
            DropColumns("a", "b"),
            MapColumnValues("c", {"hello": 1, "world": 2}),
        )
        data = pipeline(data)

In the example above, the "a" and "b" columns is dropped and the values of "c" column is transformed into 1 and 2.
The advantage of pipeline pattern is that the pipelines can be easily added, removed, reordered without changing the
preprocessing procedure thoroughly.
"""

import queue
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from typing_extensions import override

# The NLTK toolkit needs downloading before usage. Calling download every time is unnecessary, so here we first try
# to import it, and download the data if the LookupError occurs.
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
    "ProgressiveVectorizationByTfidf",
    "PairedVectorizationByTfidf",
]

# The stopwords provided by NLTK. Transformed into a set (i.e. HashSet in other languages) to keep the words appear
# once.
_stopwords = set(stopwords.words("english"))


class PreprocessPipeline(ABC):
    """
    The base class of all preprocess pipelines. It receives a DataFrame and does some transformations,
    and then returns the preprocessed data.
    """

    @abstractmethod
    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        pass


class Sequential(PreprocessPipeline):
    """
    Aggregate a pipeline sequence into a single pipeline. Execute them in their definition order when called.
    """

    _pipelines: list[PreprocessPipeline]

    def __init__(self, *pipelines: PreprocessPipeline) -> None:
        self._pipelines = list(pipelines)

    @override
    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        with tqdm(total=len(self._pipelines), leave=False) as progress:
            for pipeline in self._pipelines:
                progress.set_description(f"on pipeline {type(pipeline).__name__}")
                frame = pipeline(frame)
                progress.update()
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
    Execute a sequence of ColumnPreprocessPipeline on several specific columns. Execute them in their definition
    order when called on a specific column of a DataFrame.
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
        with tqdm(total=len(self._columns), leave=False) as column_progress:
            for column in self._columns:
                column_progress.set_description(f"on column {column}")
                with tqdm(total=len(self._pipelines), leave=False) as pipeline_progress:
                    for pipeline in self._pipelines:
                        pipeline_progress.set_description(f"on pipeline {type(pipeline).__name__}")
                        frame = pipeline(frame, column)
                        pipeline_progress.update()
                column_progress.update()
        return frame


def _column_pipeline_of(f):
    """
    Generate a Pipeline class from a computation function.

    This function is defined for convenience. For example, we can get the number of words in a text by calling
    _word_count() on it, and create a new column according to the counted values.

        Example:

            word_count = len("some text".split())
            dataframe["word_count"] = word_count

    Similarly, we can create a "length" column by calling len() on that text.

        Example:

            length = len("some text")
            dataframe["length"] = length

    And this boilerplate is redundant and boring. More importantly, it is not a Pipeline, which can be added,
    removed and reordered in practice.

    This function receives a function, which receives an element of a specific column, and returns the corresponding
    value of a new column. The new column is named after the original column and the function itself.
    """
    new_column_suffix = f.__name__

    @override
    class InlinePipeline(ColumnPreprocessPipeline):
        def __call__(self, frame: pd.DataFrame, column: str) -> pd.DataFrame:
            frame[f"{column}{new_column_suffix}"] = frame[column].apply(f)
            return frame

    return InlinePipeline


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


# The generated inline pipelines.
#
# For example, the ComputeLength is a pipeline generated from the _length function, which calls _length() on every
# element of a specific column, and create a new column named "{column}_length".
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
    """
    Also an inline-pipeline generator function.

    For more details, see doc comments about _column_pipeline_of().
    """

    class InlinePipeline(ColumnPreprocessPipeline):
        @override
        def __call__(self, frame: pd.DataFrame, column: str) -> pd.DataFrame:
            frame[f"{column}_{new}"] = frame[f"{column}_{numerator}"] / frame[f"{column}_{denominator}"]
            return frame

    return InlinePipeline


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


@dataclass
class ProgressiveVectorizationByTfidf(ColumnPreprocessPipeline):
    """
    A text-vectorization pipeline.

    This pipeline receives a TfidfVectorizer during its lifetime, and, if created with fit_transform=True,
    perform fit_transform() on text columns every time it is called.

    It is called "Progressive" because the vectorizer is not reset: it's parameter is continuously updated with
    different text columns applied. This behavior may influence the training procedure, and thus the final accuracy.
    """

    fit_transform: bool
    vectorizer: TfidfVectorizer

    def __call__(self, frame: pd.DataFrame, column: str) -> pd.DataFrame:
        if self.fit_transform:
            vectorized_columns = self.vectorizer.fit_transform(frame[column])
        else:
            vectorized_columns = self.vectorizer.transform(frame[column])
        series = pd.DataFrame(
            vectorized_columns.toarray(),
            columns=[f"tfidf_{column}_{i}" for i in range(vectorized_columns.shape[1])],
        )
        frame = pd.concat([frame, series], axis=1)
        return frame


class PairedVectorizationByTfidf(ColumnPreprocessPipeline):
    """
    A text-vectorization pipeline, with each column a brand new TfidfVectorizer.

    This pipeline receives a vectorizer_queue when created. It creates a new TfidfVectorizer and put it into the
    queue when called with fit_transform=True, and get an existing TfidfVectorizer out of the queue when
    fit_transform=False.

    This pipeline ensures the vectorizers are independent of the others from different text columns. This is the
    adopted one in the legacy code.
    """

    _vectorizer_queue: queue.Queue
    _fit_transform: bool
    _args: tuple
    _kwargs: dict

    def __init__(self, vectorizer_queue: queue.Queue, fit_transform: bool, *args, **kwargs) -> None:
        self._vectorizer_queue = vectorizer_queue
        self._fit_transform = fit_transform
        self._args = args
        self._kwargs = kwargs

    def __call__(self, frame: pd.DataFrame, column: str) -> pd.DataFrame:
        if self._fit_transform:
            vectorizer = TfidfVectorizer(*self._args, **self._kwargs)
            vectorized_columns = vectorizer.fit_transform(frame[column])
            self._vectorizer_queue.put(vectorizer)
        else:
            vectorizer = self._vectorizer_queue.get()
            vectorized_columns = vectorizer.transform(frame[column])
        series = pd.DataFrame(
            vectorized_columns.toarray(),
            columns=[f"tfidf_{column}_{i}" for i in range(vectorized_columns.shape[1])],
        )
        frame = pd.concat([frame, series], axis=1)
        return frame
