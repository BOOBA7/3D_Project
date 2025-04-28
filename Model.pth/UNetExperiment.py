"""
This module represents a UNet experiment and contains a class that handles the experiment lifecycle.

The class follows a typical UNet experiment workflow for segmentation tasks, and the basic lifecycle is as follows:

run(): This method runs for each epoch, calling train() and validate() for training and validation respectively.

train(): A single epoch training on the training dataset.

validate(): Evaluation on the validation dataset with additional metrics like loss, sensitivity, specificity, ground truth sum, and likelihood ratio.

test(): Inference on the test set.

Additional evaluation methods like sensitivity, specificity, ground truth sum, and likelihood ratio have been implemented to enrich the validation process. These metrics provide deeper insights into model performance beyond the basic loss, helping in more comprehensive evaluation during the validation phase.
"""


"""
This module represents a UNet experiment and contains a class that handles
the experiment lifecycle
"""
import os
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#from data_prep.SlicesDataset import SlicesDataset
#from utils.utils import log_to_tensorboard
#from utils.volume_stats import Dice3d, Jaccard3d
#from networks.RecursiveUNet import UNet
#from inference.UNetInferenceAgent import UNetInferenceAgent

class UNetExperiment:
    """
    This class implements the basic life cycle for a segmentation task with UNet (https://arxiv.org/abs/1505.04597).
    The basic life cycle of a UNetExperiment is:

        run():
            for epoch in n_epochs:
                train()
                validate()
        test()
    """
    def __init__(self, config, split, dataset):
        self.n_epochs = config.n_epochs
        self.split = split
        self._time_start = ""
        self._time_end = ""
        self.epoch = 0
        self.name = config.name

        # Create output folders
        dirname = f'{time.strftime("%Y-%m-%d_%H%M", time.gmtime())}_{self.name}'
        self.out_dir = os.path.join(config.test_results_dir, dirname)
        os.makedirs(self.out_dir, exist_ok=True)

        # Create data loaders
        self.train_loader = DataLoader(SlicesDataset(dataset[split["train"]]),
                batch_size=config.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(SlicesDataset(dataset[split["val"]]),
                batch_size=config.batch_size, shuffle=True, num_workers=0)

        self.test_data = dataset[split["test"]]

        # Set up device
        if not torch.cuda.is_available():
            print("WARNING: No CUDA device is found. This may take significantly longer!")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up model
        self.model = UNet(num_classes=3)
        self.model.to(self.device)

        # Set up optimizer and learning rate scheduler
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

        # Set up TensorBoard logging
        self.tensorboard_train_writer = SummaryWriter(comment="_train")
        self.tensorboard_val_writer = SummaryWriter(comment="_val")

    def train(self):
        """
        Train for a single epoch on the training dataset.
        """
        print(f"Training epoch {self.epoch}...")
        self.model.train()

        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            data = batch["image"].to(self.device)
            target = batch["seg"].long().to(self.device)

            prediction = self.model(data)
            prediction_softmax = F.softmax(prediction, dim=1)

            loss = self.loss_function(prediction, target[:, 0, :, :])
            loss.backward()
            self.optimizer.step()

            if (i % 10) == 0:
                print(f"\nEpoch: {self.epoch} Train loss: {loss}, {100*(i+1)/len(self.train_loader):.1f}% complete")
                counter = 100*self.epoch + 100*(i/len(self.train_loader))

                log_to_tensorboard(
                    self.tensorboard_train_writer,
                    loss,
                    data,
                    target,
                    prediction_softmax,
                    prediction,
                    counter)

            print(".", end='')

        print("\nTraining complete")

    def validate(self):
        """
        Evaluate on the validation dataset, return the average loss and other custom metrics.
        """
        print(f"Validating epoch {self.epoch}...")

        self.model.eval()
        loss_list = []
        sensitivity_list = []
        specificity_list = []
        likelihood_ratio_list = []
        ground_truth_sum_list = []

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                data = batch["image"].to(self.device)
                target = batch["seg"].long().to(self.device)

                prediction = self.model(data)
                prediction_softmax = F.softmax(prediction, dim=1)

                loss = self.loss_function(prediction, target[:, 0, :, :])
                loss_list.append(loss.item())

                prediction_bin = (prediction_softmax.argmax(dim=1) > 0).cpu().numpy()
                target_bin = target.cpu().numpy()

                for b in range(data.shape[0]):
                    report = full_additional_report(prediction_bin[b], target_bin[b], sample_id=i)
                    sensitivity_list.append(report["Sensitivity"])
                    specificity_list.append(report["Specificity"])
                    likelihood_ratio_list.append(report["Likelihood_Ratio"])
                    ground_truth_sum_list.append(report["Ground_Truth_Sum"])

                print(f"Batch {i}. Data shape {data.shape} Loss {loss.item()}")

        avg_loss = np.mean(loss_list)
        avg_sensitivity = np.mean(sensitivity_list)
        avg_specificity = np.mean(specificity_list)
        avg_likelihood_ratio = np.mean(likelihood_ratio_list)
        avg_ground_truth_sum = np.mean(ground_truth_sum_list)

        self.scheduler.step(avg_loss)

        log_to_tensorboard(
            self.tensorboard_val_writer,
            avg_loss,
            data,
            target,
            prediction_softmax,
            prediction,
            (self.epoch+1) * 100,
            sensitivity=avg_sensitivity,
            specificity=avg_specificity,
            likelihood_ratio=avg_likelihood_ratio,
            ground_truth_sum=avg_ground_truth_sum
        )

        print(f"Validation complete")

    def save_model_parameters(self):
        """
        Save model parameters to a file.
        """
        path = os.path.join(self.out_dir, "model.pth")
        torch.save(self.model.state_dict(), path)

    def load_model_parameters(self, path=''):
        """
        Load model parameters from a file.
        """
        if not path:
            model_path = os.path.join(self.out_dir, "model.pth")
        else:
            model_path = path

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        else:
            raise Exception(f"Could not find path {model_path}")

    def run_test(self):
        """
        Run inference on the test set.
        """
        print("Testing...")
        self.model.eval()

        inference_agent = UNetInferenceAgent(model=self.model, device=self.device)

        out_dict = {}
        out_dict["volume_stats"] = []
        dc_list = []
        jc_list = []

        for i, x in enumerate(self.test_data):
            pred_label = inference_agent.single_volume_inference(x["image"])

            dc = Dice3d(pred_label, x["seg"])
            jc = Jaccard3d(pred_label, x["seg"])
            dc_list.append(dc)
            jc_list.append(jc)

            out_dict["volume_stats"].append({
                "filename": x['filename'],
                "dice": dc,
                "jaccard": jc
            })
            print(f"{x['filename']} Dice {dc:.4f}. {100*(i+1)/len(self.test_data):.2f}% complete")

        out_dict["overall"] = {
            "mean_dice": np.mean(dc_list),
            "mean_jaccard": np.mean(jc_list)
        }

        print("\nTesting complete.")
        return out_dict

    def run(self):
        """
        Run the full training/validation cycle.
        """
        self._time_start = time.time()

        print("Experiment started.")

        for self.epoch in range(self.n_epochs):
            self.train()
            self.validate()

        self.save_model_parameters()

        self._time_end = time.time()
        print(f"Run complete. Total time: {time.strftime('%H:%M:%S', time.gmtime(self._time_end - self._time_start))}")
