import torch
from tqdm import tqdm

from dommel_library.datastructs import TensorDict
from dommel_library.train import Trainer
from dommel_library.train.losses import Log



class ModelTrainer(Trainer):
    def __init__(self, model, train_dataset, loss, optimizer, log_dir,
                 warmup=1, **kwargs):
        # divide loss by the sequence length to have normalized loss value
        # sequence_length = train_dataset.sample(1).shape[1]
        # for _, loss_dict in loss._losses.items():
        #     weight = loss_dict.get("weight", 1.0)
        #     weight /= sequence_length
        #     loss_dict["weight"] = weight


        Trainer.__init__(self, model, train_dataset, loss,
                         optimizer, log_dir, **kwargs)
        self._warmup = warmup

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
            loss = self._loss(output_dict,output_dict) #this line is different from dommel trainer
            loss.backward()
            self._clip(self._model.parameters())
            self._optimizer.step()
            self._loss.post_backprop()
            logs += self._loss.logs
        return logs

    # def _visualize_prior(self, batch):
    #     with torch.no_grad():

    #         sequence = self._model(batch)
    #     prior_images = visualize_sequence(sequence, **self._vis_args)
    #     return prior_images

    # def _log_visualization(self, train_logs, val_logs, epoch):
    #     train_logs, val_logs = super()._log_visualization(train_logs, val_logs, epoch)

    #     # add prior visualizations
    #     prior_images = self._visualize_prior(self._visualize_batch["train"])
    #     for k, img in prior_images.items():
    #         train_logs.add("query/" + k, img, "image")
           

    #     if self._val_loader:
    #         prior_images = self._visualize_prior(self._visualize_batch["val"])
    #         for k, img in prior_images.items():
    #             train_logs.add("query/" + k, img, "image")
                

    #    return train_logs, val_logs


    def _initial_log(self, start_epoch):
        # compute initial batch and log this
        # also visualize ground truth
        #difference with dommel, output_dict is used for sequence visualisation
        with torch.no_grad():
            train_logs = Log()
            output_dict = self._model(self._visualize_batch["train"])
            _ = self._loss(output_dict, self._visualize_batch["train"])
            train_logs += self._loss.logs

            # visualize ground truth as well
            # dataset_images = visualize_sequence(
            #     output_dict, **self._vis_args, **self.vis_mapping
            # )
            # for k, img in dataset_images.items():
            #     train_logs.add("ground_truth/" + k, img, "image")

            val_logs = None
            if self._val_loader:
                self._model.eval()
                val_logs = Log()
                output_dict = self._model(self._visualize_batch["val"])
                _ = self._loss(output_dict, self._visualize_batch["val"])
                val_logs += self._loss.logs

                # dataset_images = visualize_sequence(
                #     output_dict, **self._vis_args, **self.vis_mapping
                # )
                # for k, img in dataset_images.items():
                #     val_logs.add("ground_truth/" + k, img, "image")

            self._log_callback(train_logs, val_logs, start_epoch)