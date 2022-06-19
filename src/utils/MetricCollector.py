import logging
from typing import Dict, List, Tuple

import numpy as np

from src.project_enums import EnumMetricParams
from src.utils import utils


class MetricCollector:
    """Collect metrics for later use and check if net is still improving"""

    def __init__(self, params: List[Dict]):
        """ if you want to use load_state_dict() just pass an empty dict for params here

        Args:
            params: [{name:string, mode: string<min|max> = 'min', use_for_is_best=False, use_for_plateau_detection=False, patience=5, min_delta=0, percentage=False}]
        """
        self.log = logging.getLogger("MetricCollector")

        self.metrics = {}
        for param in params:
            name = param['name']
            del param['name']

            param.setdefault(EnumMetricParams.mode, 'min')
            param.setdefault(EnumMetricParams.use_for_is_best, False)
            param.setdefault(EnumMetricParams.use_for_plateau_detection, False)
            param.setdefault(EnumMetricParams.patience, 5)
            param.setdefault(EnumMetricParams.min_delta, 0)
            param.setdefault(EnumMetricParams.percentage, False)

            assert param['mode'] in ['min', 'max']

            self.metrics[name] = {
                **param,
                EnumMetricParams.values: [],
                EnumMetricParams.best: None,
                EnumMetricParams.unchanged_for_epochs: 0
            }

    def step(self, metrics) -> Tuple[bool, bool]:
        """

        Args:
            metrics: {name: value, name2: value2, ...}

        Returns:

        """
        assert sorted(metrics.keys()) == sorted(self.metrics.keys())

        if utils.is_dist_avail_and_initialized():
            self.log.debug("syncing test results between gpus")
            self.log.debug(f"before: {metrics}")
            metrics = utils.reduce_dict(metrics)
            self.log.debug(f"after : {metrics}")

        model_has_improved = False
        for name, value in metrics.items():
            assert name in self.metrics
            assert not np.isnan(value)

            self.metrics[name][EnumMetricParams.values].append(value)
            if self.metrics[name][EnumMetricParams.best] is None or self._is_value_better(name, value):
                self.metrics[name][EnumMetricParams.best] = value
                self.metrics[name][EnumMetricParams.unchanged_for_epochs] = 0
                if self.metrics[name][EnumMetricParams.use_for_is_best]:
                    model_has_improved = True
            else:
                self.metrics[name][EnumMetricParams.unchanged_for_epochs] += 1
        return model_has_improved, self._is_plateau()

    def state_dict(self):
        return self.metrics

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            for required_param in EnumMetricParams:
                if required_param not in v.keys():
                    raise AssertionError(f'"{required_param}" not in state_dict')  # assert does not give useful error messages
        self.metrics = state_dict

    def get_latest_values(self):
        """get a dict with a summary of the current state eg for logging"""
        summary = {}
        for name, values in self.metrics.items():
            assert len(values[EnumMetricParams.values]) > 0
            summary[name] = {'best': values[EnumMetricParams.best],
                             'unchanged_for_epochs': values[EnumMetricParams.unchanged_for_epochs],
                             'last_value': values[EnumMetricParams.values][-1]}
        return summary

    def get_last_value(self, key):
        return self.metrics[key][EnumMetricParams.values][-1]

    def _is_value_better(self, name, value) -> bool:
        """check if value of name is better than best previous value"""
        delta = self.metrics[name][EnumMetricParams.min_delta]
        if self.metrics[name][EnumMetricParams.percentage]:
            delta *= self.metrics[name][EnumMetricParams.best] / 100

        if self.metrics[name][EnumMetricParams.mode] == 'min':
            return value < self.metrics[name][EnumMetricParams.best] - delta
        return value > self.metrics[name][EnumMetricParams.best] + delta

    def _is_plateau(self) -> bool:
        """early stopping will trigger if patience was hit for each param with use_for_plateau_detection==True"""
        for name, values in self.metrics.items():
            if values[EnumMetricParams.use_for_plateau_detection] \
                    and values[EnumMetricParams.unchanged_for_epochs] < values[EnumMetricParams.patience]:
                # inverse logic: if criterion is not hit for one attribute we'll continue learning
                return False
        return True

    def __getitem__(self, key) -> Dict:
        return self.metrics[key]
