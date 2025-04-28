"""
This script sets up and runs the training and testing processes for a UNet model. Key steps include:

Configuration: The Config class holds all the necessary parameters (e.g., learning rate, batch size, epochs, etc.).

Data Loading: The script loads the hippocampus dataset, reshapes it, and creates training, validation, and test splits.

Experiment Setup: An experiment instance (UNetExperiment) is created using the configuration and dataset splits.

Training and Testing: The model is trained, and test results are evaluated.

Results Saving: The results, including configuration details, are saved as a JSON file.

This pipeline is designed to streamline the process of training and evaluating the UNet model on medical image data.
"""

"""
This file contains code that will kick off training and testing processes
"""
import os
import json

from networks.RecursiveUNet import UNet

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = "/content/clean_dataset"  # <-- REMPLACER PAR TON DOSSIER RÉEL
        self.n_epochs = 10
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = "/content/clean_dataset/result"  # <-- REMPLACER PAR TON DOSSIER RÉEL

if __name__ == "__main__":
    # Get configuration
    c = Config()

    # Load data
    print("Loading data...")
    data = LoadHippocampusData(c.root_dir, y_shape=c.patch_size, z_shape=c.patch_size)

    # Create test-train-val split
    keys = list(range(len(data)))

    # Shuffle keys for randomness (important for fair training)
    import random
    random.seed(42)
    random.shuffle(keys)

    split = dict()
    split["train"] = keys[:int(0.7 * len(keys))]
    split["val"] = keys[int(0.7 * len(keys)):int(0.85 * len(keys))]
    split["test"] = keys[int(0.85 * len(keys)):]  # Remaining 15%

    print("Dataset split:", {k: len(v) for k, v in split.items()})

    # Set up and run experiment
    exp = UNetExperiment(c, split, data)

    # Run training
    exp.run()

    # Run testing
    results_json = exp.run_test()

    results_json["config"] = vars(c)

    # Save results
    os.makedirs(exp.out_dir, exist_ok=True)
    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))

