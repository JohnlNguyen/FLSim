#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""This file runs cifar10 partioned using dirichlet.

The dataset must be created beforehand using this notebook https://fburl.com/anp/d8nxmsc3.

  Typical usage example:

  buck run papaya/toolkit/simulation/baselines:run_cifar_dirichlet -- --config-file \\
  fbcode/fblearner/flow/projects/papaya/examples/hydra_configs/cifar10_dirichlet.json
"""
from typing import Dict, NamedTuple

import flsim.configs
import hydra  # @manual
import torch
import torchvision.models as models
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config
from hydra.utils import instantiate  # @manual
from omegaconf import DictConfig, OmegaConf
from opacus.validators import ModuleValidator
from torchvision import transforms
from torchvision.models.squeezenet import SqueezeNet, squeezenet1_0
import math
import random
from typing import Any, Dict, Generator, Iterable, Iterator, List, Tuple

import numpy as np
import torch
from flsim.data.data_provider import IFLDataProvider, IFLUserData
from flsim.data.data_sharder import FLDataSharder
from flsim.interfaces.data_loader import IFLDataLoader
from flsim.utils.data.data_utils import batchify
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from tqdm import tqdm
from flsim.utils.example_utils import (
    DataLoader,
    DataProvider,
    SimpleConvNet,
    FLModel,
    MetricsReporter,
)
import os
from torchvision.datasets.cifar import CIFAR10


class LEAFDataLoader(IFLDataLoader):
    SEED = 2137
    random.seed(SEED)

    def __init__(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int,
        drop_last: bool = False,
    ):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def fl_train_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.train_dataset, self.drop_last)

    def fl_eval_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.eval_dataset, drop_last=False)

    def fl_test_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.test_dataset, drop_last=False)

    def _batchify(
        self, dataset: Dataset, drop_last=False
    ) -> Generator[Dict[str, Generator], None, None]:
        # pyre-fixme[16]: `Dataset` has no attribute `__iter__`.
        for one_user_inputs, one_user_labels in dataset:
            data = list(zip(one_user_inputs, one_user_labels))
            random.shuffle(data)
            one_user_inputs, one_user_labels = zip(*data)
            batch = {
                "features": batchify(one_user_inputs, self.batch_size, drop_last),
                "labels": batchify(one_user_labels, self.batch_size, drop_last),
            }
            yield batch


class LEAFUserData(IFLUserData):
    def __init__(self, user_data: Dict[str, Generator], eval_split):
        self._user_batches = []
        self._eval_batches = []
        self._num_train_batches = 0
        self._num_train_examples = 0
        self._eval_split = eval_split

        self._num_eval_batches = 0
        self._num_eval_examples = 0
        user_features = list(user_data["features"])
        user_labels = list(user_data["labels"])
        total = sum(len(batch) for batch in user_labels)

        for features, labels in zip(user_features, user_labels):
            if self._num_eval_examples < int(total * self._eval_split):
                self._num_eval_batches += 1
                self._num_eval_examples += LEAFUserData.get_num_examples(
                    labels)
                self._eval_batches.append(
                    LEAFUserData.fl_training_batch(features, labels)
                )
            else:
                self._num_train_batches += 1
                self._num_train_examples += LEAFUserData.get_num_examples(
                    labels)
                self._user_batches.append(
                    LEAFUserData.fl_training_batch(features, labels)
                )

    def train_data(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterator to return a user batch data
        """
        for batch in self._user_batches:
            yield batch

    def eval_data(self):
        for batch in self._eval_batches:
            yield batch

    def num_train_batches(self):
        return self._num_train_batches

    def num_eval_batches(self):
        return self._num_eval_batches

    def num_train_examples(self) -> int:
        """
        Returns the number of examples
        """
        return self._num_train_examples

    def num_eval_examples(self):
        return self._num_eval_examples

    @staticmethod
    def get_num_examples(batch: List) -> int:
        return len(batch)

    @staticmethod
    def fl_training_batch(
        features: List[torch.Tensor], labels: List[float]
    ) -> Dict[str, torch.Tensor]:
        return {
            "features": torch.stack(features),
            "labels": torch.Tensor(np.array(labels)),
        }


class LEAFDataProvider(IFLDataProvider):
    def __init__(self, data_loader, eval_split=0.1):
        self.data_loader = data_loader
        self._train_users = self._create_fl_users(
            data_loader.fl_train_set(), eval_split=0.0
        )
        self._eval_users = self._create_fl_users(
            data_loader.fl_eval_set(), eval_split)
        self._test_users = self._create_fl_users(
            data_loader.fl_test_set(), eval_split)

    def train_user_ids(self) -> List[int]:
        return list(self._train_users.keys())

    def num_train_users(self) -> int:
        return len(self._train_users)

    def get_train_user(self, user_index: int) -> IFLUserData:
        if user_index in self._train_users:
            return self._train_users[user_index]
        else:
            raise IndexError(
                f"Index {user_index} is out of bound for list with len {self.num_train_users()}"
            )

    def train_users(self) -> Iterable[IFLUserData]:
        for user_data in self._train_users.values():
            yield user_data

    def eval_users(self) -> Iterable[IFLUserData]:
        for user_data in self._eval_users.values():
            yield user_data

    def test_users(self) -> Iterable[IFLUserData]:
        for user_data in self._test_users.values():
            yield user_data

    def _create_fl_users(
        self, iterator: Iterator, eval_split
    ) -> Dict[int, IFLUserData]:
        return {
            user_index: LEAFUserData(user_data, eval_split)
            for user_index, user_data in tqdm(
                enumerate(iterator), desc="Creating FL User", unit="user"
            )
        }


def set_random_seed(seed_value, use_cuda: bool = True) -> None:
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    os.environ["PYTHONHASHSEED"] = str(seed_value)  # Python hash buildin
    if use_cuda:
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


class CIFAROutput(NamedTuple):
    log_dir: str
    eval_scores: Dict[str, float]
    test_scores: Dict[str, float]


def build_data_provider(data_config, drop_last: bool = False):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]
    )
    with open(data_config.train_file, 'rb') as f:
        train_dataset = torch.load(f)
    test_dataset = CIFAR10("/data/home/ngjhn/FLSim/examples/data/",
                           train=False, download=True, transform=transform)

    data_loader = LEAFDataLoader(
        train_dataset,
        [list(zip(*test_dataset))],
        [list(zip(*test_dataset))],
        data_config.local_batch_size,
    )
    data_provider = LEAFDataProvider(data_loader)
    print(f"Clients in total: {data_provider.num_train_users()}")
    return data_provider


def train(
    trainer_config,
    model_config,
    data_config,
    use_cuda_if_available: bool = True,
    fb_info=None,
    world_size: int = 1,
    rank: int = 0,
) -> CIFAROutput:
    data_provider = build_data_provider(data_config)

    for seed in range(model_config.num_trials):
        cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
        set_random_seed(seed, use_cuda=cuda_enabled)
        metrics_reporter = MetricsReporter(
            [Channel.TENSORBOARD, Channel.STDOUT],
            target_eval=model_config.target_eval,
            window_size=model_config.window_size,
            average_type=model_config.average_type,
            log_dir=data_config.log_dir,
        )
        print("Created metrics reporter")

        device = torch.device(f"cuda:{rank}" if cuda_enabled else "cpu")
        print(f"Training launched on device: {device}")

        if model_config.pretrained:
            if model_config.use_resnet:
                model = models.resnet18(pretrained=True)
                model = ModuleValidator.fix(model)
            else:
                model = squeezenet1_0(pretrained=True)
        else:
            if model_config.use_resnet:
                model = models.resnet18()
                model = ModuleValidator.fix(model)
            else:
                model = SqueezeNet(num_classes=10)

        # pyre-fixme[6]: Expected `Optional[str]` for 2nd param but got `device`.
        global_model = FLModel(model, device)
        if cuda_enabled:
            global_model.fl_cuda()

        trainer = instantiate(
            trainer_config, model=global_model, cuda_enabled=cuda_enabled
        )
        print(f"Created {trainer_config._target_}")

        final_model, eval_score = trainer.train(
            data_provider=data_provider,
            metric_reporter=metrics_reporter,
            num_total_users=data_provider.num_train_users(),
            distributed_world_size=world_size,
            rank=rank,
        )
        test_metric = trainer.test(
            data_provider=data_provider,
            metric_reporter=MetricsReporter(
                [Channel.STDOUT], log_dir=data_config.log_dir),
        )
        if eval_score[MetricsReporter.ACCURACY] <= model_config.target_eval:
            break
    return CIFAROutput(
        log_dir=metrics_reporter.writer.log_dir,
        # pyre-fixme[61]: `eval_score` is undefined, or not always defined.
        eval_scores=eval_score,
        # pyre-fixme[61]: `test_metric` is undefined, or not always defined.
        test_scores=test_metric,
    )


@hydra.main(config_path=None, config_name="cifar10_single_process")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    trainer_config = cfg.trainer
    model_config = cfg.model
    data_config = cfg.data

    train(
        trainer_config=trainer_config,
        data_config=data_config,
        model_config=model_config,
        use_cuda_if_available=True,
        fb_info=None,
        world_size=1,
    )
    if cfg.cluster:
        checkpoint_name = f"{args.optim}_server-lr_{args.server_lr}_client-lr_{args.client_lr}_seed_{args.seed}_pretrained_{args.pretrained}"
        # executor is the submission interface (logs are dumped in the folder)
        executor = submitit.AutoExecutor(folder="stackoverflow_logs")

        # set parameters for running the job
        num_gpus_per_node = 1
        nodes = 1
        executor.update_parameters(
            name=checkpoint_name,
            gpus_per_node=num_gpus_per_node,
            tasks_per_node=1,  # one task per GPU
            cpus_per_task=10,  # 10 cpus per gpu is generally good
            nodes=nodes,
            # Below are cluster dependent parameters
            slurm_partition="a100",
            slurm_time=3000,  # 3000 mins
        )
        job = executor.submit(
            train, args
        )
        print(job.job_id, checkpoint_name)  # ID of your job

if __name__ == "__main__":
    cfg = maybe_parse_json_config()
    run(cfg)
