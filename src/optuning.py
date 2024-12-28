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
    """
    Optimize an OptunableProblemSolver with optuna.

    The basic idea is to call solve() method with different params generated on each trial. Since the return value of a
    ProblemSolver will always be a ProblemSolution class, we'll just use the accuracy attribute in that object as our
    target, and use optuna to maximize it.

    This function only consumes the solver object and the number of trials. Additional arguments will be passed onto
    the prepare_parameter() method of the concrete solver class.

    The best accuracy is returned as a float value. Moreover, both the best accuracy and the params will be written
    in "optuned-parameters.toml" under the project root dir, which is convenient to load in and applied to model when
    submit to Kaggle.
    """

    optuna.logging.disable_default_handler()  # the default logging of optuna is annoying.

    # We'll use the tqdm package to create a relative silent progressbar instead of printing everything out. The
    # index of trials is used as the progress and the accuracy of each trial is shown as the postfix of the progressbar.
    with tqdm(total=n_trials, leave=False, desc="Optuna") as progressbar:
        # In this case, we just tell optuna to maximize the accuracy.
        study = optuna.create_study(direction="maximize")

        # This callback is set to update the progressbar after each trial.
        def update_progressbar_callback(_, frozen_trial: FrozenTrial) -> None:
            progressbar.set_postfix({"acc": frozen_trial.value})
            progressbar.update()

        # This matches optuna's requirement: a callable receiving only the Trial object, and returns the metric (in
        # this case, a float representing the accuracy).
        #
        # We just pass the Trial object and additional args onto the prepare_parameter() class method to generate a
        # Params object and call solve() on that. ProblemSolution.accuracy is returned finally.
        def optimize_step(trial: Trial) -> float:
            params = solver.__class__.prepare_parameter(trial, *args, **kwargs)
            solution = solver.solve(params)
            return solution.accuracy

        # Now we've defined everything, just let optuna optimize it.
        study.optimize(optimize_step, n_trials, callbacks=[update_progressbar_callback], gc_after_trial=True)

    # After optimization, the accuracy and parameters will be cached, in TOML format.
    if not _PARAM_CACHE_PATH.exists():
        param_cache = {}
    else:
        param_cache = toml.load(_PARAM_CACHE_PATH)

    # And each model has its own section in the TOML file. The accuracy and params will only be updated when the
    # freshly optimized one is better than that already written in the cache file.
    solver_name = solver.__class__.__name__
    if solver_name in param_cache and param_cache[solver_name]["accuracy"] >= study.best_value:
        return param_cache[solver_name]["accuracy"]
    param_cache[solver_name] = {
        "accuracy": study.best_value,
        "params": study.best_params,
    }
    with open(_PARAM_CACHE_PATH, "w", encoding="utf-8") as cache:
        # noinspection PyTypeChecker
        toml.dump(param_cache, cache)

    # Finally, the best accuracy is returned.
    return study.best_value
