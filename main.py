import warnings

import pandas as pd
import toml

from src.solvers.lgbm import LGBMSolver, LGBMParams


def load_settings() -> dict:
    with open("environment-settings.toml", encoding="utf-8") as file:
        settings = toml.load(file)

    enabled = settings["paths"]["enabled"]
    found = False
    for key, value in settings["paths"].items():
        if key == enabled:
            settings["paths"] = value
            found = True
            break
    assert found, "invalid path settings"

    return settings


CONFIG = load_settings()


def main() -> None:
    with warnings.catch_warnings():  # warnings in preprocessing stage are ignored.
        warnings.simplefilter("ignore")
        solver = LGBMSolver(
            CONFIG["paths"]["train"],
            CONFIG["paths"]["test"],
            target_column="winner",
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
        "early_stop": 40,
        "random_state": 42,
    }

    predictions = solver.solve(LGBMParams(**params)).predictions

    sample = pd.read_csv(CONFIG["paths"]["sample"])
    sample["winner"] = predictions
    sample["winner"] = sample["winner"].map({0: "model_a", 1: "model_b"})

    sample.to_csv("submission_refactored.csv", index=False)


# Guideline recommended Main Guard
if __name__ == "__main__":
    main()
