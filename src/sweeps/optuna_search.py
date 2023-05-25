from typing import List, Union

import optuna

from .config.cfg import SweepConfig
from .params import HyperParameterSet, HyperParameter
from .run import SweepRun, RunState


def _to_optuna_state(run: SweepRun):
    if run.state == RunState.finished:
        return optuna.trial.TrialState.COMPLETE
    if run.state == RunState.crashed:
        return optuna.trial.TrialState.FAIL
    if run.state == RunState.failed:
        return optuna.trial.TrialState.FAIL
    if run.state == RunState.killed:
        return optuna.trial.TrialState.FAIL
    if run.state == RunState.preempted:
        return optuna.trial.TrialState.FAIL
    if run.state == RunState.preempting:
        return optuna.trial.TrialState.FAIL
    if run.state == RunState.pending:
        return optuna.trial.TrialState.WAITING
    if run.state == RunState.running:
        return optuna.trial.TrialState.RUNNING
    assert False, "Never reach here."


def _to_optuna_distribution(param: HyperParameter) -> optuna.distributions.BaseDistribution:
    if param.type == HyperParameter.CATEGORICAL:
        return optuna.distributions.CategoricalDistribution(
            choices=param.config["values"]
        )
    if param.type == HyperParameter.INT_UNIFORM:
        return optuna.distributions.IntDistribution(
            low=param.config["min"], high=param.config["max"],
        )
    if param.type == HyperParameter.UNIFORM:
        return optuna.distributions.FloatDistribution(
            low=param.config["min"], high=param.config["max"],
        )
    raise ValueError("Unsupported HyperParameter type.")


def _to_optuna_distributions(params: HyperParameterSet):
    distributions = {}
    for param in HyperParameterSet:
        distributions[param.name] = _to_optuna_distribution(param)
    return distributions


def _to_optuna_params(run: SweepRun, params: HyperParameterSet):
    params = {}
    for param in HyperParameterSet:
        params[param.name] = run.config[param.name]["value"]
    return params


def optuna_search_next_runs(
    runs: List[SweepRun],
    sweep_config: Union[dict, SweepConfig],
    validate: bool = False,
    n: int = 1,
) -> List[SweepRun]:

    if validate:
        sweep_config = SweepConfig(sweep_config)

    if sweep_config["method"] != "bayes":
        raise ValueError("Invalid sweep configuration for optuna_search_next_runs")
    
    # Get search space.
    params = HyperParameterSet.from_config(sweep_config["parameters"])
    distributions = _to_optuna_distributions(params)

    # Create Optuna study
    study = optuna.create_study(
        storage=None,
        study_name=None,
        direction=sweep_config["metric"]["goal"],
    )

    # Report existing runs to Optuna.
    metric_kind = "maximum" if study.direction == optuna.study.StudyDirection.MAXIMIZE else "minimum"
    metric_name = sweep_config["metric"]["name"]
    for run in runs:
        metric = run.metric_extremum(metric_name, kind=metric_kind)
        study.add_trial(
            optuna.create_trial(
                state=_to_optuna_state(run),
                value=run.metric_extremum(metric_name, kind=metric_kind),
                params=_to_optuna_params(run),
                distributions=distributions,
                value=metric,
            )
        )

    # Apply Optuna's Ask-and-Tell to suggest params and report metric values.
    retval = []
    for _ in range(n):
        trial = study.ask(distributions)
        for param in params:
            param.value = trial.params[param.name]
        run = SweepRun(config=params.to_config())
        metric = run.metric_extremum(metric_name, kind=metric_kind)
        study.tell(trial, values=[metric])
        retval.append(run)

    return retval
