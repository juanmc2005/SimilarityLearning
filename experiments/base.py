import argparse
from os.path import join

import common
from core.base import Trainer
from core.plugins.logging import TrainLogger, MetricFileLogger
from core.plugins.storage import ModelLoader
from datasets.base import SimDatasetPartition
from models import SimNet
from losses.config import LossConfig


class ModelEvaluationExperiment:

    def evaluate_on_dev(self, plot: bool) -> float:
        raise NotImplementedError

    def evaluate_on_test(self) -> float:
        raise NotImplementedError


class TrainingExperiment:

    def __init__(self, task: str, nfeat: int):
        self.task = task
        self.nfeat = nfeat
        self.dataset = None

    def _parse_args(self):
        parser = common.get_arg_parser()
        self._on_parser(parser)
        return parser.parse_args()

    def _load_dataset(self, loss: str, batch_size: int):
        print(f"[Task: {self.task.upper()}]")
        print(f"[Loss: {loss.upper()}]")
        print('[Loading Dataset...]')
        self.dataset, nclass = self._create_dataset(loss, batch_size)
        train = self.dataset.training_partition()
        print('[Dataset Loaded]')
        return train, nclass

    def _logging(self, interval: int, log_path: str, train: SimDatasetPartition, train_plugins: list, test_plugins: list):
        if interval in range(1, 101):
            print(f"[Logging: {common.enabled_str(True)} (every {interval}%)]")
            test_plugins.append(MetricFileLogger(log_path=join(log_path, f"metric.log")))
            train_plugins.append(TrainLogger(interval, train.nbatches(),
                                             loss_log_path=join(log_path, f"loss.log")))
            self._on_logging_enabled(train_plugins, test_plugins)
        else:
            print(f"[Logging: {common.enabled_str(False)}]")

    def _model_saving(self, save: bool, train_plugins: list, test_plugins: list):
        print(f"[Model Saving: {common.enabled_str(save)}]")
        if save:
            self._on_saving_enabled(train_plugins, test_plugins)

    def _plotting(self, plot: bool, train_plugins: list, test_plugins: list):
        print(f"[Plotting: {common.enabled_str(plot)}]")
        if plot:
            self._on_plotting_enabled(train_plugins, test_plugins)

    def train(self):
        # Script arguments
        args = self._parse_args()
        # Set custom seed before doing anything
        common.set_custom_seed(args.seed)
        # Create directory to save plots, models, results, etc
        log_path = common.create_log_dir(args.exp_id, args.task, args.loss)
        print(f"Logging to {log_path}")
        # Load dataset
        train, nclass = self._load_dataset(args.loss, args.batch_size)
        # Create model
        config = common.get_config(args.loss, self.nfeat, nclass, self.task,
                                   args.margin, args.triplet_strategy, args.semihard_negatives)
        model = self._create_model(config)
        # Train and evaluation plugins
        test_plugins = []
        train_plugins = []
        # Logging configuration
        self._logging(args.log_interval, log_path, train, train_plugins, test_plugins)
        # Model saving configuration
        self._model_saving(args.save, train_plugins, test_plugins)
        # Plotting configuration
        self._plotting(args.plot, train_plugins, test_plugins)
        # Other useful plugins
        self._on_all_plugins_added(train_plugins, test_plugins)
        # Evaluation configuration
        train_plugins.append(self._create_evaluator(args.loss, args.batch_size, config, test_plugins))
        # Training configuration
        trainer = Trainer(args.loss, model, config.loss, train, config.optimizer(model, self.task, args.lr),
                          model_loader=ModelLoader(args.recover) if args.recover is not None else None,
                          callbacks=train_plugins)
        print(f"[LR: {args.lr}]")
        print(f"[Batch Size: {args.batch_size}]")
        print(f"[Epochs: {args.epochs}]")
        print()
        # Start training
        trainer.train(args.epochs)

    def _on_parser(self, parser: argparse.ArgumentParser):
        pass

    def _create_dataset(self, loss: str, batch_size: int) -> tuple:
        raise NotImplementedError

    def _create_model(self, config: LossConfig) -> SimNet:
        raise NotImplementedError

    def _on_logging_enabled(self, train_plugins: list, test_plugins: list):
        pass

    def _on_saving_enabled(self, train_plugins: list, test_plugins: list):
        pass

    def _on_plotting_enabled(self, train_plugins: list, test_plugins: list):
        pass

    def _on_all_plugins_added(self, train_plugins: list, test_plugins: list):
        pass

    def _create_evaluator(self, loss: str, batch_size: int, config: LossConfig, test_plugins: list):
        raise NotImplementedError
