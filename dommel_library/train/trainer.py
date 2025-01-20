import pathlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dommel_library.train.losses import Log
from dommel_library.datastructs import Dict, TensorDict
from dommel_library.modules.visualize import visualize_sequence

from tqdm import tqdm
from numbers import Number

import logging
import numpy as np

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        loss,
        optimizer,
        log_dir,
        val_dataset=None,
        scheduler=None,
        batch_size=10,
        device="cpu",
        log_epoch=1,
        log_debug=False,
        save_epoch=500,
        vis_epoch=None,
        vis_args=None,
        vis_batch_size=None,
        num_workers=0,
        verbose=True,
        clip_grad_norm=None,
        logging_platform="tensorboard",
        wandb_entity="dommel_library",
        experiment_config=None,
        ** kwargs,
    ):
        """
        Create a Trainer object
        :param model: The model to optimize.
        :param train_dataset: The train dataset to optimize on.
        Must be dommel Dataset.
        :param loss: The loss function to optimize.
        Must be a dommel LossAggregate.
        :param optimizer: The Pytorch optimizer.
        :param log_dir: Path to store the logs and models.
        :param val_dataset: Optionally, a validation dataset.
        :param scheduler: Optionally, a learning rate scheduler.
        :param batch_size: The mini-batch size for iterating the train set.
        :param device: The device to load the tensors and model
        on during training.
        :param log_epoch: After each how many epochs to log the loss.
        :param log_debug: Flag indicating whether to log grad and weight
        statistics for debugging.
        :param save_epoch: After each how many epochs to save the model.
        :param vis_epoch: After each how many epochs to log visualizations.
        This can be single number (i.e. similar to log_epoch), or a dict like
        {"before_rate": 1, "n": 10, "after_rate": 10}. In this example before
        the n=10th epoch every epoch is visualized, and after that each 10th.
        This to mitigate the overhead of extensive visualization in the logs.
        Default behavior is to log each log_epoch first 10 and then visualize
        10 times less frequent.
        :param vis_batch_size: Optionally, a different batch size for the
        batch that is visualized in the logs.
        :param num_workers: The number of workers of the DataLoaders
        :param verbose: Flag indicating whether to print tqdm.
        :param clip_grad_norm: Gradient clipping value, no gradient clipping
        when value is None
        :param logging_platform: Platform to use for logging the progress of
        training. Options are ["tensorboard", "wandb"]
        :param wandb_entity: Wandb entity/team
        :param experiment_config: experiment config dict to store in wandb
        :param kwargs: Other arguments.
        """
        self._device = device

        self._model = model.to(self._device)
        self._loss = loss.to(self._device)
        self._optimizer = optimizer
        self._scheduler = scheduler

        self._train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )

        # Create directories for logging
        log_dir = pathlib.Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        self._model_dir = log_dir / "models"
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._optimizer_dir = log_dir / "optimizers"
        self._optimizer_dir.mkdir(parents=True, exist_ok=True)

        self._log_epoch = log_epoch
        self._log_debug = log_debug
        if isinstance(vis_epoch, dict):
            self._vis_epoch = Dict(vis_epoch)
        elif isinstance(vis_epoch, Number):
            self._vis_epoch = Dict(
                {"before_rate": vis_epoch, "n": 0, "after_rate": vis_epoch}
            )
        else:
            self._vis_epoch = Dict(
                {
                    "before_rate": self._log_epoch,
                    "n": self._log_epoch * 10,
                    "after_rate": self._log_epoch * 10,
                }
            )

        self._save_epoch = save_epoch

        self._vis_args = vis_args
        if self._vis_args is None:
            self._vis_args = dict()

        if not vis_batch_size:
            vis_batch_size = batch_size
        else:
            vis_batch_size = min(batch_size, vis_batch_size)
        self._visualize_batch = dict()
        self._visualize_batch["train"] = TensorDict(
            next(iter(self._train_loader))
        ).to(self._device)[0:vis_batch_size, ...]

        self._val_loader = None
        if val_dataset is not None:
            self._val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=True,
            )
            self._visualize_batch["val"] = TensorDict(
                next(iter(self._val_loader))
            ).to(self._device)[0:vis_batch_size, ...]

        self._tensorboard_logging = logging_platform == "tensorboard"
        if not self._tensorboard_logging:
            try:
                import wandb
                run = log_dir.name
                project = log_dir.parent.name
                wandb.init(project=project, entity=wandb_entity,
                           name=run, config=experiment_config)
                wandb.watch(self._model)
                self._log_callback = self._wandb_log_callback
            except Exception as err:
                # revert to tensorboard logging
                logger.warning(
                    f"Failed to initialize wandb: {err}, "
                    f"reverting to tensorboard logging.")
                self._tensorboard_logging = True

        if self._tensorboard_logging:
            train_log_dir = log_dir / "train"
            train_log_dir.mkdir(parents=True, exist_ok=True)
            self._train_writer = SummaryWriter(log_dir=train_log_dir)
            if self._val_loader is not None:
                val_log_dir = log_dir / "val"
                val_log_dir.mkdir(parents=True, exist_ok=True)
                self._val_writer = SummaryWriter(log_dir=val_log_dir)

            self._log_callback = self._tensorboard_log_callback

        self._verbose = verbose

        # Allow for gradient clipping if desired
        if clip_grad_norm is not None:
            self._clip = lambda x: nn.utils.clip_grad_norm_(
                x, max_norm=clip_grad_norm
            )
        else:
            # Avoid the if-check in _epoch -> map it to a no-op function
            self._clip = lambda x: x

    def _epoch(self):
        """
        Execute a single epoch of training
        :return: Log object containing information of the epoch step
        """
        logs = Log()
        self._model.train()
        for input_dict in tqdm(self._train_loader, disable=not self._verbose):
            input_dict = TensorDict(input_dict).to(self._device)
            self._optimizer.zero_grad()
            output_dict = self._model(input_dict)
            loss = self._loss(output_dict, input_dict)
            loss.backward()
            self._clip(self._model.parameters())
            self._optimizer.step()
            self._loss.post_backprop()
            logs += self._loss.logs
        return logs

    def _validate(self):
        """
        Execute a single pass over the validation set
        :return: Log object containing information of the validation step
        """
        logs = Log()
        self._model.eval()
        with torch.no_grad():
            for input_dict in tqdm(
                self._val_loader, disable=not self._verbose
            ):
                input_dict = TensorDict(input_dict).to(self._device)
                output_dict = self._model(input_dict)
                _ = self._loss(output_dict, input_dict)
                logs += self._loss.logs
        return logs

    def _save_model_callback(self, epoch):
        """
        Is called every epoch to store the model if epoch is a multiple of
        self._save_epoch
        :param epoch: current epoch during training
        """
        if epoch % self._save_epoch == 0:
            torch.save(
                self._model.state_dict(),
                self._model_dir / "model-{:04d}.pt".format(epoch),
            )
            torch.save(
                self._optimizer.state_dict(),
                self._optimizer_dir / "optim-{:04d}.pt".format(epoch),
            )
            if self._scheduler:
                torch.save(
                    self._scheduler.state_dict(),
                    self._optimizer_dir / "scheduler-{:04d}.pt".format(epoch),
                )

    def _log_visualization(self, train_logs, val_logs, epoch):
        """
        This method is called right before logging to tensorboard. This is
        where a user can create custom visualizations and add those to the
        log.
        :param train_logs:
        :param val_logs:
        :return: train_logs and val_logs with the new logs
        """
        with torch.no_grad():
            train_output = self._model(self._visualize_batch["train"])

        pred_images = visualize_sequence(train_output, **self._vis_args)
        for k, img in pred_images.items():
            train_logs.add("prediction/" + k, img, "image")

        if self._val_loader:
            with torch.no_grad():
                val_output = self._model(self._visualize_batch["val"])

            pred_images = visualize_sequence(val_output, **self._vis_args)
            for k, img in pred_images.items():
                val_logs.add("prediction/" + k, img, "image")

        return train_logs, val_logs

    def _tensorboard_log_callback(
        self, train_logs=None, val_logs=None, epoch=0
    ):
        """
        Generic tensorboard logger.
        Receives a dictionary with key: ( value, type )
        :param train_logs: logs should be written as above
        :param val_logs: logs should be written as above
        :param epoch: epoch number
        """
        vis_epoch = (
            self._vis_epoch.before_rate
            if epoch < self._vis_epoch.n
            else self._vis_epoch.after_rate
        )
        if epoch % vis_epoch == 0:
            # Get visualization on eval-batch
            train_logs, val_logs = self._log_visualization(
                train_logs, val_logs, epoch
            )

        if epoch % self._log_epoch == 0:
            if train_logs is not None:
                train_logs.to_writer(self._train_writer, epoch)
            if val_logs is not None:
                val_logs.to_writer(self._val_writer, epoch)

            # Visualize grads and values of the parameters
            if self._log_debug:
                for name, p in self._model.named_parameters():
                    self._train_writer.add_histogram(
                        f"{name.replace('.','/')}/value", p, epoch
                    )
                    self._train_writer.add_histogram(
                        f"{name.replace('.','/')}/grad", p, epoch
                    )

            # Visualize learning rate if there is a scheduler
            if self._scheduler:
                for param_group in self._optimizer.param_groups:
                    self._train_writer.add_scalar(
                        "learning_rate/", param_group["lr"], epoch
                    )

    def _wandb_log_callback(self, train_logs=None, val_logs=None, epoch=0):
        import wandb

        vis_epoch = (
            self._vis_epoch.before_rate
            if epoch < self._vis_epoch.n
            else self._vis_epoch.after_rate
        )
        if epoch % vis_epoch == 0:
            # Get visualization on eval-batch
            train_logs, val_logs = self._log_visualization(
                train_logs, val_logs, epoch
            )

        wandb_logs = {}
        if epoch % self._log_epoch == 0:
            if train_logs is not None:
                wandb_logs.update(train_logs.to_wandb(prefix="train"))
            if val_logs is not None:
                wandb_logs.update(val_logs.to_wandb(prefix="val"))

        # Visualize learning rate if there is a scheduler
        if self._scheduler:
            for param_group in self._optimizer.param_groups:
                wandb_logs.update(
                    {f"learning_rate/{param_group['lr']}": epoch}
                )

        wandb_logs["epoch"] = epoch
        wandb.log(wandb_logs)

    def _initial_log(self, start_epoch):
        # compute initial batch and log this
        # also visualize ground truth
        with torch.no_grad():
            train_logs = Log()
            output_dict = self._model(self._visualize_batch["train"])
            _ = self._loss(output_dict, self._visualize_batch["train"])
            train_logs += self._loss.logs

            # visualize ground truth as well
            dataset_images = visualize_sequence(
                self._visualize_batch["train"], **self._vis_args
            )
            for k, img in dataset_images.items():
                train_logs.add("ground_truth/" + k, img, "image")

            val_logs = None
            if self._val_loader:
                self._model.eval()
                val_logs = Log()
                output_dict = self._model(self._visualize_batch["val"])
                _ = self._loss(output_dict, self._visualize_batch["val"])
                val_logs += self._loss.logs

                dataset_images = visualize_sequence(
                    self._visualize_batch["val"], **self._vis_args
                )
                for k, img in dataset_images.items():
                    val_logs.add("ground_truth/" + k, img, "image")

            self._log_callback(train_logs, val_logs, start_epoch)

    def _schedule_callback(self, train_logs, val_logs):
        """
        Callback function for scheduling the optimizer learning rate
        :param train_logs: training logs can be used to determine new lr value
        :param val_logs: validation logs can be used to determine new lr value
        """
        if self._scheduler:
            self._scheduler.step()

    def load(self, checkpoint_epoch):
        """
        Load model and trainer
        :param checkpoint_epoch: Epoch from which to restore
        """
        model_path = self._model_dir / (f"model-{checkpoint_epoch:04d}.pt")

        models = [m for m in self._model_dir.glob("*.pt")]
        if model_path not in models:
            logger.warning(
                f"Model epoch is not stored, is it a multiple "
                f"of {self._save_epoch}?"
            )
            model_path = sorted(models)[-1]

        # load model weights
        model_state_dict = torch.load(model_path)
        self._model.load_state_dict(model_state_dict)

        # load optimizer weights
        optim_path = sorted([o for o in self._optimizer_dir.glob("*.pt")])[-1]
        optim_state_dict = torch.load(optim_path)
        self._optimizer.load_state_dict(optim_state_dict)

        # load scheduler weights
        if self._scheduler:
            scheduler_state_dict = torch.load(
                str(optim_path).replace("optim", "scheduler")
            )
            self._scheduler.load_state_dict(scheduler_state_dict)
        return int(str(model_path).split("-")[-1][:-3])

    def train(self, num_epochs, start_epoch=0):
        """
        Training loop
        :param num_epochs: number of epochs to train
        :param start_step: The step to start at, relevant for continuing
            training
        :return: nothing
        """
        # load a saved model
        if start_epoch != 0:
            start_epoch = self.load(start_epoch)

        # do initial logging (i.e. visualizing ground truth etc.)
        self._initial_log(start_epoch)

        # Do the train loop
        for epoch in range(start_epoch + 1, num_epochs + 1):
            logger.info(f"{epoch}/{num_epochs}")
            train_logs = self._epoch()
            logger.info(
                "Train loss: {}".format(
                    np.array(train_logs.logs["Loss/loss"]["value"])
                    .flatten()
                    .mean(),
                )
            )

            val_logs = None
            if self._val_loader:
                val_logs = self._validate()
                logger.info(
                    "Validation loss: {}".format(
                        np.array(val_logs.logs["Loss/loss"]["value"])
                        .flatten()
                        .mean(),
                    )
                )

            self._schedule_callback(train_logs, val_logs)
            self._log_callback(train_logs, val_logs, epoch)
            self._save_model_callback(epoch)
