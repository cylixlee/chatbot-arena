import nltk

nltk.download("stopwords")

# === Starter ===


import numpy as np
import pandas as pd


from tqdm import tqdm


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import string
import warnings

warnings.filterwarnings("ignore")
pd.options.display.max_columns = None

SEED = 42
n_splits = 5

# === Preprocess ===

"""
Clone the ABD module file.

Due to the limitation of Kaggle, modules can only be referenced through "input". Input 
files have their own path, which is not good for importing.
"""

# sp = '/kaggle/input/abdbase/AbdML/main.py'
# tp = '/kaggle/working/main.py'
#
# with open(sp, 'r', encoding='utf-8') as file:
#     content = file.read()
# with open(tp, 'w', encoding='utf-8') as file:
#     file.write(content)

from src.abdml import AbdBase

"""
Load the data provided by Kaggle.

The path is different between Kaggle virtual environments and the local one. Thus, 
a configuration is adopted and those paths can be switched conveniently.
"""
import toml

CONFIG = toml.load("environment-settings.toml")
PATHS = CONFIG["paths"][CONFIG["paths"]["adopted"]]

train = pd.read_parquet(PATHS["train"])
test = pd.read_parquet(PATHS["test"])
sample = pd.read_csv(PATHS["sample"])

train["winner"] = train["winner"].map({"model_a": 0, "model_b": 1})
drop_cols = ["model_a", "model_b", "language", "scored"]

train = train.drop(columns=drop_cols, errors="ignore")
test = test.drop(columns=drop_cols, errors="ignore")

train["id"] = train["id"].astype("category")
test["id"] = test["id"].astype("category")

stop_words = set(stopwords.words("english"))


def text_stat(df, txt_col):
    for col in tqdm(txt_col, desc="Processing text columns"):
        df[f"{col}_length"] = df[col].apply(len)
        df[f"{col}_word_count"] = df[col].apply(lambda x: len(x.split()))
        df[f"{col}_char_count"] = df[col].apply(
            lambda x: sum([len(word) for word in x.split()])
        )
        df[f"{col}_avg_word_length"] = df[f"{col}_char_count"] / df[f"{col}_word_count"]

        df[f"{col}_punctuation_count"] = df[col].apply(
            lambda x: sum([1 for char in x if char in string.punctuation])
        )
        df[f"{col}_capitalized_count"] = df[col].apply(
            lambda x: sum([1 for word in x.split() if word.isupper()])
        )
        df[f"{col}_special_char_count"] = df[col].apply(
            lambda x: sum(
                [1 for char in x if not char.isalnum() and not char.isspace()]
            )
        )
        df[f"{col}_stopwords_count"] = df[col].apply(
            lambda x: len([word for word in x.split() if word.lower() in stop_words])
        )
        df[f"{col}_unique_word_count"] = df[col].apply(lambda x: len(set(x.split())))
        df[f"{col}_lexical_diversity"] = (
            df[f"{col}_unique_word_count"] / df[f"{col}_word_count"]
        )

        df[f"{col}_word_length_mean"] = df[col].apply(
            lambda x: np.mean([len(word) for word in x.split()])
        )
        df[f"{col}_word_length_median"] = df[col].apply(
            lambda x: np.median([len(word) for word in x.split()])
        )
        df[f"{col}_word_length_max"] = df[col].apply(
            lambda x: max([len(word) for word in x.split()], default=0)
        )
        df[f"{col}_word_length_min"] = df[col].apply(
            lambda x: min([len(word) for word in x.split()], default=0)
        )

        df[f"{col}_sentence_length_mean"] = df[col].apply(
            lambda x: np.mean(
                [len(sentence.split()) for sentence in x.split(".") if sentence.strip()]
            )
        )
        df[f"{col}_sentence_length_median"] = df[col].apply(
            lambda x: np.median(
                [len(sentence.split()) for sentence in x.split(".") if sentence.strip()]
            )
        )
        df[f"{col}_sentence_length_max"] = df[col].apply(
            lambda x: max(
                [
                    len(sentence.split())
                    for sentence in x.split(".")
                    if sentence.strip()
                ],
                default=0,
            )
        )
        df[f"{col}_sentence_length_min"] = df[col].apply(
            lambda x: min(
                [
                    len(sentence.split())
                    for sentence in x.split(".")
                    if sentence.strip()
                ],
                default=0,
            )
        )

    df["response_length_diff_a_b"] = df["response_a_length"] - df["response_b_length"]
    df["response_length_diff_b_a"] = df["response_b_length"] - df["response_a_length"]
    df["response_length_ratio_a_b"] = df["response_a_length"] / (
        df["response_b_length"] + 1e-6
    )
    df["response_length_ratio_b_a"] = df["response_b_length"] / (
        df["response_a_length"] + 1e-6
    )

    return df


txt_col = ["prompt", "response_a", "response_b"]

train = text_stat(train, txt_col)
test = text_stat(test, txt_col)


def tf_fe(train, test, text_columns, max_features=3000, analyzer="char_wb"):
    train_features = []
    test_features = []

    for col in tqdm(text_columns, desc="Processing text columns", unit="col"):
        vectorizer = TfidfVectorizer(analyzer=analyzer, max_features=max_features)
        train_tfidf_col = vectorizer.fit_transform(train[col])
        test_tfidf_col = vectorizer.transform(test[col])
        train_tfidf_col = pd.DataFrame(
            train_tfidf_col.toarray(),
            columns=[f"tfidf_{col}_{i}" for i in range(train_tfidf_col.shape[1])],
        )
        test_tfidf_col = pd.DataFrame(
            test_tfidf_col.toarray(),
            columns=[f"tfidf_{col}_{i}" for i in range(test_tfidf_col.shape[1])],
        )
        train_features.append(train_tfidf_col)
        test_features.append(test_tfidf_col)

    train_with_tfidf = pd.concat([train, *train_features], axis=1)
    test_with_tfidf = pd.concat([test, *test_features], axis=1)

    return train_with_tfidf, test_with_tfidf


txt_col = ["prompt", "response_a", "response_b"]
train, test = tf_fe(train, test, txt_col)

train = train.drop(columns=txt_col, errors="ignore")
test = test.drop(columns=txt_col, errors="ignore")

# === AbdBase | LGBM ===

SEED = 42

base = AbdBase(
    train_data=train,
    test_data=test,
    target_column="winner",
    gpu=False,
    problem_type="classification",
    metric="accuracy",
    seed=SEED,
    n_splits=5,
    early_stop=True,
    num_classes=2,
    test_prob=True,
    fold_type="SKF",
    weights=None,
    tf_vec=False,
)

Params = {
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

meanOFFL, meanTestL, *_ = base.Train_ML(Params, "LGBM", e_stop=40)

# === Submission ===

sample["winner"] = np.round(meanTestL).astype("int")
sample["winner"] = sample["winner"].map({0: "model_a", 1: "model_b"})

sample.to_csv("submission.csv", index=False)
sample.head()
