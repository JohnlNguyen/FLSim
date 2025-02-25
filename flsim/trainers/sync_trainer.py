#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from time import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from flsim.channels.message import Message
from flsim.clients.base_client import Client
from flsim.clients.dp_client import DPClientConfig, DPClient
from flsim.common.timeline import Timeline
from flsim.data.data_provider import IFLDataProvider
from flsim.interfaces.metrics_reporter import IFLMetricsReporter, Metric, TrainingStage
from flsim.interfaces.model import IFLModel
from flsim.servers.sync_dp_servers import SyncDPSGDServerConfig
from flsim.servers.sync_secagg_servers import SyncSecAggServerConfig
from flsim.servers.sync_servers import (
    ISyncServer,
    SyncServerConfig,
    FedAvgOptimizerConfig,
)
from flsim.trainers.trainer_base import FLTrainer, FLTrainerConfig
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.config_utils import is_target
from flsim.utils.distributed.fl_distributed import FLDistributedUtils
from flsim.utils.fl.common import FLModelParamUtils
from flsim.utils.fl.stats import RandomVariableStatsTracker
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm


class SyncTrainer(FLTrainer):
    """Implements FederatedAveraging: https://arxiv.org/abs/1602.05629

    Attributes:
        epochs (int): Training epochs
        report_train_metrics (bool): Whether metrics on training data should be
            computed and reported.
    """

    def __init__(
        self,
        *,
        model: IFLModel,
        cuda_enabled: bool = False,
        **kwargs,
    ):
        init_self_cfg(
            self,
            # pyre-fixme[10]: Name `__class__` is used but not defined.
            component_class=__class__,
            config_class=SyncTrainerConfig,
            **kwargs,
        )

        super().__init__(model=model, cuda_enabled=cuda_enabled, **kwargs)
        self.server: ISyncServer = instantiate(
            # pyre-ignore[16]
            self.cfg.server,
            global_model=model,
            channel=self.channel,
        )
        self.clients = {}

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        if OmegaConf.is_missing(cfg.server, "_target_"):
            cfg.server = SyncServerConfig(optimizer=FedAvgOptimizerConfig())

    def global_model(self) -> IFLModel:
        """This function makes it explicit that self.global_model() is owned
        by the server, not by SyncTrainer
        """
        return self.server.global_model

    @property
    def is_user_level_dp(self):
        return is_target(self.cfg.server, SyncDPSGDServerConfig)

    @property
    def is_sample_level_dp(self):
        return is_target(self.cfg.client, DPClientConfig)

    @property
    def is_secure_aggregation_enabled(self):
        return is_target(self.cfg.server, SyncSecAggServerConfig)

    def create_or_get_client_for_data(self, dataset_id: int, datasets: Any):
        """This function is used to create clients in a round. Thus, it
        is called UPR * num_rounds times per training run. Here, we use
        <code>OmegaConf.structured</code> instead of <code>hydra.instantiate</code>
        to minimize the overhead of hydra object creation.
        """
        if self.is_sample_level_dp:
            client = DPClient(
                # pyre-ignore[16]
                **OmegaConf.structured(self.cfg.client),
                dataset=datasets.get_train_user(dataset_id),
                name=f"client_{dataset_id}",
                timeout_simulator=self._timeout_simulator,
                store_last_updated_model=self.cfg.report_client_metrics,
                channel=self.channel,
                cuda_manager=self._cuda_state_manager,
            )
        else:
            client = Client(
                **OmegaConf.structured(self.cfg.client),
                dataset=datasets.get_train_user(dataset_id),
                name=f"client_{dataset_id}",
                timeout_simulator=self._timeout_simulator,
                store_last_updated_model=self.cfg.report_client_metrics,
                channel=self.channel,
                cuda_manager=self._cuda_state_manager,
            )
        self.clients[dataset_id] = client
        return self.clients[dataset_id]

    def train(
        self,
        data_provider: IFLDataProvider,
        metric_reporter: IFLMetricsReporter,
        num_total_users: int,
        distributed_world_size: int,
        rank: int = 0,
    ) -> Tuple[IFLModel, Any]:
        """Train and eval a model, the model states will be modified. This function
        iterates over epochs specified in config, and for each epoch:

            1. Trains model in a federated way: different models are trained over data
                from different users, and are averaged into 'model' at the end of epoch
            2. Evaluate averaged model using evaluation data
            3. Calculate metrics based on evaluation results and select best model

        Args:
            data_provider (IFLDataProvider): provide training, evaluation, and test data
                iterables and get a user's data based on user ID
            metric_reporter (IFLMetricsReporter): compute metric based on training
                output and report results to console, file, etc.
            num_total_users (int): number of total users for training

        Returns:
            model, best_metric: the trained model together with the best metric

        Note:
            one `epoch` = go over all users once is not True here
            since users in each round are selected randomly, this isn't precisely true
            we may go over some users more than once, and some users never
            however, as long as users_per_round << num_total_users, this will work
            the alternative is to keep track of all users that have already
            been selected in the current epoch - impractical and not worth it
            however, we may have that option in simulation later on.
            TODO correct note if above option added.
        """
        # set up synchronization utilities for distributed training
        FLDistributedUtils.setup_distributed_training(
            distributed_world_size, use_cuda=self.cuda_enabled
        )  # TODO do not call distributed utils here, this is upstream responsibility

        if rank != 0:
            FLDistributedUtils.suppress_output()

        # pyre-fixme[16]: `SyncTrainer` has no attribute `cfg`.
        assert self.cfg.users_per_round % distributed_world_size == 0

        best_metric = None
        best_model_state = self.global_model().fl_get_module().state_dict()
        # last_best_epoch = 0
        users_per_round = min(self.cfg.users_per_round, num_total_users)

        self.data_provider = data_provider
        num_rounds_in_epoch = self.rounds_in_one_epoch(
            num_total_users, users_per_round)
        num_users_on_worker = data_provider.num_train_users()

        # torch.multinomial requires int instead of float, cast it as int
        users_per_round_on_worker = int(
            users_per_round / distributed_world_size)
        self._validate_users_per_round(
            users_per_round_on_worker, num_users_on_worker)

        # main training loop
        num_int_epochs = math.ceil(self.cfg.epochs)
        for epoch in tqdm(
            range(1, num_int_epochs + 1), desc="Epoch", unit="epoch", position=0
        ):
            for round in tqdm(
                range(1, num_rounds_in_epoch + 1),
                desc="Round",
                unit="round",
                position=0,
            ):
                timeline = Timeline(
                    epoch=epoch, round=round, rounds_per_epoch=num_rounds_in_epoch
                )

                clients = self._client_selection(
                    num_users_on_worker,
                    users_per_round_on_worker,
                    data_provider,
                    self.global_model(),
                    epoch,
                )

                self._train_one_round(
                    timeline=timeline,
                    clients=clients,
                    users_per_round=users_per_round,
                    metric_reporter=metric_reporter
                    if self.cfg.report_train_metrics
                    else None,
                )

                # report training success rate and training time variance
                if rank == 0:
                    (best_metric, best_model_state,) = self._maybe_run_evaluation(
                        timeline=timeline,
                        data_provider=data_provider,
                        metric_reporter=metric_reporter,
                        best_metric=best_metric,
                        best_model_state=best_model_state,
                    )

                if self.stop_fl_training(
                    epoch=epoch, round=round, num_rounds_in_epoch=num_rounds_in_epoch
                ):
                    break

            if self.stop_fl_training(
                epoch=epoch,
                # pyre-fixme[61]: `round` may not be initialized here.
                round=round,
                num_rounds_in_epoch=num_rounds_in_epoch,
            ):
                break

        if rank == 0 and best_metric is not None:
            self._save_model_and_metrics(self.global_model(), best_model_state)

        return self.global_model(), best_metric

    def stop_fl_training(self, *, epoch, round, num_rounds_in_epoch) -> bool:
        # stop if necessary number of steps/epochs are completed in case of fractional epochs
        # or if client times out
        global_round_num = (epoch - 1) * num_rounds_in_epoch + round
        return (
            (global_round_num / num_rounds_in_epoch)
            # pyre-fixme[16]: `SyncTrainer` has no attribute `cfg`.
            >= self.cfg.epochs
            or self._timeout_simulator.stop_fl()
        )

    def _drop_overselected_users(
        self, clents_triggered: List[Client], num_users_keep: int
    ) -> List[Client]:
        """
        sort users by their training time, and only keep num_users_keep users
        """
        all_training_times = [c.get_total_training_time()
                              for c in clents_triggered]
        all_training_times.sort()
        # only select first num_users_keep userids sort by their finish time
        num_users_keep = min([num_users_keep, len(all_training_times)])
        last_user_time = all_training_times[num_users_keep - 1]
        num_users_added = 0
        clients_used = []
        for c in clents_triggered:
            # if two clients finished at the same time, order for entering
            # the cohort is arbitrary
            if (c.get_total_training_time() <= last_user_time) and (
                num_users_added < num_users_keep
            ):
                num_users_added += 1
                clients_used.append(c)

        return clients_used

    def _client_selection(
        self,
        num_users: int,
        users_per_round: int,
        data_provider: IFLDataProvider,
        epoch: int,
    ) -> List[Client]:
        # pyre-fixme[16]: `SyncTrainer` has no attribute `cfg`.
        num_users_overselected = math.ceil(
            users_per_round / self.cfg.dropout_rate)
        user_indices_overselected = self.server.select_clients_for_training(
            num_total_users=num_users,
            users_per_round=num_users_overselected,
            data_provider=data_provider,
            epoch=epoch,
        )
        clients_triggered = [
            self.create_or_get_client_for_data(i, self.data_provider)
            for i in user_indices_overselected
        ]
        clients_to_train = self._drop_overselected_users(
            clients_triggered, users_per_round
        )
        return clients_to_train

    def _save_model_and_metrics(self, model: IFLModel, best_model_state):
        model.fl_get_module().load_state_dict(best_model_state)

    def _train_one_round(
        self,
        timeline: Timeline,
        clients: Iterable[Client],
        users_per_round: int,
        metric_reporter: Optional[IFLMetricsReporter],
    ) -> None:
        self.server.init_round()
        for client in clients:
            client_delta, weight = client.generate_local_update(
                self.global_model(), metric_reporter
            )
            message = Message(client_delta, weight)
            self.server.receive_update_from_client(message)
        self.server.step()

        self._report_train_metrics(
            model=self.global_model(),
            timeline=timeline,
            metric_reporter=metric_reporter,
        )
        self._evaluate_global_model_after_aggregation(
            model=self.global_model(),
            timeline=timeline,
            users_per_round=users_per_round,
            metric_reporter=metric_reporter,
        )
        self._calc_post_epoch_communication_metrics(
            timeline,
            metric_reporter,
        )

    def _calc_privacy_metrics(
        self, clients: Iterable[Client],
    ) -> List[Metric]:
        """
        Calculates privacy metrics.
        """

        metrics = []
        if self.is_user_level_dp:
            user_eps = self.server.privacy_budget.epsilon  # pyre-fixme
            metrics.append(Metric("user level dp (eps)", user_eps))
        if self.is_sample_level_dp:
            # calculate sample level dp privacy loss statistics.
            all_client_eps = torch.Tensor(
                [c.privacy_budget.epsilon for c in clients]  # pyre-fixme
            )
            mean_client_eps = all_client_eps.mean()
            max_client_eps = all_client_eps.max()
            min_client_eps = all_client_eps.min()
            p50_client_eps = torch.median(all_client_eps)
            sample_dp_metrics: List[Metric] = Metric.from_args(
                mean=mean_client_eps,
                min=min_client_eps,
                max=max_client_eps,
                median=p50_client_eps,
            )
            metrics.append(Metric("sample level dp (eps)", sample_dp_metrics))

        return metrics

    def _calc_overflow_metrics(
        self,
        clients: Iterable[Client],
        model: IFLModel,
        users_per_round: int,
        metric_reporter: Optional[IFLMetricsReporter],
    ) -> List[Metric]:
        """
        Calculates overflow metrics.
        """
        metrics = []
        if self.is_secure_aggregation_enabled:
            for client in clients:
                client.eval(model=model, metric_reporter=metric_reporter)
            (
                convert_overflow_perc,
                aggregate_overflow_perc,
            ) = self.server.calc_avg_overflow_percentage(  # pyre-fixme
                users_per_round, model
            )
            overflow_metrics: List[Metric] = Metric.from_args(
                convert_overflow_percentage=convert_overflow_perc,
                aggregate_overflow_percentage=aggregate_overflow_perc,
            )
            metrics.append(Metric("overflow per round", overflow_metrics))

        return metrics

    def _evaluate_global_model_after_aggregation(
        self,
        clients: Iterable[Client],
        model: IFLModel,
        timeline: Timeline,
        users_per_round: int,
        metric_reporter: Optional[IFLMetricsReporter] = None,
    ):
        if (
            metric_reporter is not None
            # pyre-fixme[16]: `SyncTrainer` has no attribute `cfg`.
            and self.cfg.report_train_metrics
            and self.cfg.report_train_metrics_after_aggregation
            and timeline.tick(1.0 / self.cfg.train_metrics_reported_per_epoch)
        ):
            with torch.no_grad():
                model.fl_get_module().eval()
                for eval_user in self.data_provider.eval_users():
                    for batch in eval_user.eval_data():
                        batch_metrics = model.get_eval_metrics(batch)
                        if metric_reporter is not None:
                            metric_reporter.add_batch_metrics(batch_metrics)
                model.fl_get_module().train()

            print(f"Reporting {timeline} for aggregation")
            privacy_metrics = self._calc_privacy_metrics(
                clients, model, metric_reporter
            )
            overflow_metrics = self._calc_overflow_metrics(
                clients, model, users_per_round, metric_reporter
            )

            metric_reporter.report_metrics(
                model=model,
                reset=True,
                stage=TrainingStage.AGGREGATION,
                timeline=timeline,
                epoch=timeline.global_round_num(),  # for legacy
                print_to_channels=True,
                extra_metrics=privacy_metrics + overflow_metrics,
            )

    def _validate_users_per_round(
        self, users_per_round_on_worker: int, num_users_on_worker: int
    ):
        assert users_per_round_on_worker <= num_users_on_worker, (
            "Users per round is greater than number of users in data provider for the worker."
            "If you are using paged dataloader, increase your num_users_per_page >> users_per_round"
        )

    @staticmethod
    def rounds_in_one_epoch(num_total_users: int, users_per_round: int) -> int:
        return math.ceil(num_total_users / users_per_round)


def force_print(is_distributed: bool, *args, **kwargs) -> None:
    if is_distributed:
        try:
            device_info = f" [device:{torch.cuda.current_device()}]"
            # pyre-fixme[28]: Unexpected keyword argument `force`.
            print(*args, device_info, **kwargs, force=True)
        except TypeError:
            pass
    else:
        print(*args, **kwargs)


@dataclass
class SyncTrainerConfig(FLTrainerConfig):
    _target_: str = fullclassname(SyncTrainer)
    server: SyncServerConfig = SyncServerConfig()
    users_per_round: int = 10
    # overselect users_per_round / dropout_rate users, only use first
    # users_per_round updates
    dropout_rate: float = 1.0
    report_train_metrics_after_aggregation: bool = False
    report_client_metrics_after_epoch: bool = False
    # Whether client metrics on eval data should be computed and reported.
    report_client_metrics: bool = False
    # how many times per epoch should we report client metrics
    # numbers greater than 1 help with plotting more precise training curves
    client_metrics_reported_per_epoch: int = 1
