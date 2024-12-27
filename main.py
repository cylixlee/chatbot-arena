import warnings

import toml

from src.optuning import optune_solver
from src.relativepath import PROJECT_DIR
from src.solvers.logistic_regression import LogisticRegressionSolver


def load_settings() -> dict:
    with open(PROJECT_DIR / "environment-settings.toml", encoding="utf-8") as file:
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
        solver = LogisticRegressionSolver(
            CONFIG["paths"]["train"],
            CONFIG["paths"]["test"],
            target_column="winner",
        )

    optune_solver(solver, n_trials=100, n_splits=5, epochs=100, random_state=42)


# Guideline recommended Main Guard
if __name__ == "__main__":
    main()
