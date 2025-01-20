import torch

from dommel_library.datastructs import cat
from dommel_library.train import Trainer
from dommel_library.modules.visualize import visualize_sequence

from experiments.OZ.models import ReplayAgent, model_rollout, imagine_rollout

class ModelTrainer(Trainer):
    def __init__(self, model, train_dataset, loss, optimizer, log_dir,
                 warmup=1, **kwargs):
        # divide loss by the sequence length to have normalized loss value
        sequence_length = train_dataset.sample(1).shape[1]
        for _, loss_dict in loss._losses.items():
            weight = loss_dict.get("weight", 1.0)
            weight /= sequence_length
            loss_dict["weight"] = weight

        Trainer.__init__(self, model, train_dataset, loss,
                         optimizer, log_dir, **kwargs)
        self._warmup = warmup

    def _visualize_prior(self, batch):
        with torch.no_grad():
            replay = ReplayAgent(batch.action[:, self._warmup:, ...])
            bootstrap = model_rollout(
                self._model, batch[:, 0:self._warmup, ...])
            prior = imagine_rollout(self._model,
                                    replay,
                                    batch.shape[1] - self._warmup)
            sequence = cat(bootstrap, prior)
        prior_images = visualize_sequence(sequence, **self._vis_args)
        return prior_images

    def _log_visualization(self, train_logs, val_logs, epoch):
        train_logs, val_logs = super()._log_visualization(train_logs, val_logs, epoch)

        # add prior visualizations
        prior_images = self._visualize_prior(self._visualize_batch["train"])
        for k, img in prior_images.items():
            train_logs.add("prior/" + k, img, "image")

        if self._val_loader:
            prior_images = self._visualize_prior(self._visualize_batch["val"])
            for k, img in prior_images.items():
                val_logs.add("prior/" + k, img, "image")

        return train_logs, val_logs
