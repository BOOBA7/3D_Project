"""
This module defines the UNetInferenceAgent class, which is responsible for running inference on 3D medical volumes using a trained UNet model. The class handles loading the model, processing input volumes, and generating predictions. It supports running inference on both padded and conformant volumes.

Key functionalities:

Initializes the model and loads parameters from a specified file path.
Runs inference on a single volume with a specified patch size.
Handles 3D volume processing by splitting it into 2D slices and performing predictions for each slice.
Supports both padded and non-padded inference.
"""


"""
Contains class that runs inferencing
"""
import torch
import numpy as np

#from networks.RecursiveUNet import UNet

#from utils.utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        
        x, y, z = volume.shape
        padded = med_reshape(volume, (x, self.patch_size, self.patch_size))
        pred = self.single_volume_inference(padded)
        return med_reshape(pred, (x, y, z))

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        slices = []

        with torch.no_grad():
            for i in range(volume.shape[0]):
                
                slice_2d = volume[i, :, :]
                input_tensor = torch.tensor(slice_2d[None, None, :, :], dtype=torch.float32).to(self.device)

                
                output = self.model(input_tensor)

                
                pred_slice = torch.argmax(output, dim=1).cpu().numpy()[0]

                slices.append(pred_slice)

        # Empiler toutes les slices pour reconstruire le volume 3D
        return np.stack(slices, axis=0)
