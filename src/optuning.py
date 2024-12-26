import optuna
import toml
from optuna import Trial
from optuna.trial import FrozenTrial
from tqdm import tqdm

from src.relativepath import PROJECT_DIR
from src.solvers import OptunableProblemSolver

__all__ = ["optune_solver"]

_PARAM_CACHE_PATH = PROJECT_DIR / "optuned-params.toml"


def optune_solver(solver: OptunableProblemSolver, n_trials: int, *args, **kwargs) -> float:
    optuna.logging.disable_default_handler()
    with tqdm(total=n_trials, leave=False, desc="Optuna") as progressbar:
        study = optuna.create_study(direction="maximize")

        def update_progressbar_callback(_, frozen_trial: FrozenTrial) -> None:
            progressbar.set_postfix({"acc": frozen_trial.value})
            progressbar.update()

        def optimize_step(trial: Trial) -> float:
            params = solver.__class__.prepare_parameter(trial, *args, **kwargs)
            solution = solver.solve(params)
            return solution.accuracy

        study.optimize(optimize_step, n_trials, callbacks=[update_progressbar_callback])

    if not _PARAM_CACHE_PATH.exists():
        param_cache = {}
    else:
        param_cache = toml.load(_PARAM_CACHE_PATH)

    solver_name = solver.__class__.__name__
    if solver_name in param_cache and param_cache[solver_name]["accuracy"] >= study.best_value:
        return param_cache[solver_name]["accuracy"]
    param_cache[solver_name] = {
        "accuracy": study.best_value,
        "params": study.best_params,
    }
    with open(_PARAM_CACHE_PATH, "w", encoding="utf-8") as cache:
        toml.dump(param_cache, cache)

    return study.best_value
