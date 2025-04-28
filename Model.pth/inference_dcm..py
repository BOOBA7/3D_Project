"""
This script performs inference on a DICOM volume, generates a report, and stores it in a clinical archive. The process involves:

Identifying relevant DICOM series and constructing a 3D NumPy volume.

Running inference to segment the hippocampus and generating a visual report with volumes and slices.

Anonymizing DICOM headers to protect patient identity and saving the report in DICOM Secondary Capture format.

Using os.command to push the report to the storage archive via an API in a medical environment.

Functions include loading DICOM files, creating the report, anonymizing headers, and interacting with the archive system.
"""

"""
Here we do inference on a DICOM volume, constructing the volume first, and then sending it to the
clinical archive

This code will do the following:
    1. Identify the series to run HippoCrop.AI algorithm on from a folder containing multiple studies
    2. Construct a NumPy volume from a set of DICOM files
    3. Run inference on the constructed volume
    4. Create report from the inference
    5. Call a shell script to push report to the storage archive
"""

import os
import sys
import datetime
import time
import shutil
import subprocess

import numpy as np
import pydicom
import torch.nn.functional as F

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from collections import Counter
from pydicom.data import get_testdata_files

from inference.UNetInferenceAgent import UNetInferenceAgent

from collections import Counter

def load_dicom_volume_as_numpy_from_list(dcmlist):
    """Loads a list of PyDicom objects into a 3D Numpy array with consistent slice shape."""

    # Count the dimensions
    shape_counts = Counter([dcm.pixel_array.shape for dcm in dcmlist])
    most_common_shape = shape_counts.most_common(1)[0][0]

    # Keep only slices with the most common shape
    valid_slices = [dcm for dcm in dcmlist if dcm.pixel_array.shape == most_common_shape]

    # Sort by instance number
    valid_slices = sorted(valid_slices, key=lambda dcm: dcm.InstanceNumber)

    if len(valid_slices) == 0:
        raise ValueError("No slices with consistent shape found.")

    # Build the volume
    slices = [np.flip(dcm.pixel_array).T for dcm in valid_slices]

    hdr = valid_slices[0]
    hdr.PixelData = None

    return (np.stack(slices, 2), hdr)



def create_report(inference, header, orig_vol, pred_vol):
    # Create a blank image for the report
    pimg = Image.new("RGB", (900, 600), "black")

    # Define the fonts
    header_font = ImageFont.load_default()
    main_font = ImageFont.load_default()

    # Initialize the drawing tool
    draw = ImageDraw.Draw(pimg)

    # Text at the top of the image
    draw.text((10, 0), "HippoVolume.AI", (255, 255, 255), font=header_font)
    draw.multiline_text((10, 90),
                        f"Patient ID: {header.PatientID}\n"
                        f"Study Date: {header.StudyDate}\n"
                        f"Series Description: {header.SeriesDescription}\n\n"
                        f"Anterior Volume: {inference['anterior']} mm³\n"
                        f"Posterior Volume: {inference['posterior']} mm³\n"
                        f"Total Volume: {inference['total']} mm³",
                        fill=(255, 255, 255), font=main_font)

    # Choose 3 slices to display: anterior, posterior, central
    slice_nums = [
        min(int(pred_vol.shape[2] * 0.25), pred_vol.shape[2] - 1),
        min(int(pred_vol.shape[2] * 0.5), pred_vol.shape[2] - 1),
        min(int(pred_vol.shape[2] * 0.75), pred_vol.shape[2] - 1)
    ]

    y_offset = 300
    for i, slc_idx in enumerate(slice_nums):
        # Ensure the index is valid for both volumes
        slc_idx = min(slc_idx, orig_vol.shape[2] - 1)  # Correct the index for orig_vol

        # Make sure the index is within bounds for both volumes
        if slc_idx >= pred_vol.shape[2] or slc_idx >= orig_vol.shape[2]:
            print(f"Index {slc_idx} is out of bounds for orig_vol with shape {orig_vol.shape} or pred_vol with shape {pred_vol.shape}")
            continue  # Skip this slice if the index is out of bounds

        # Extract the slice for both volumes
        slc = orig_vol[:, :, slc_idx]
        mask = pred_vol[:, :, slc_idx]

        # Ensure that the size of the slice and the mask is correct
        slc_img = np.flip((slc / np.max(slc)) * 255).T.astype(np.uint8)
        mask_img = np.flip(mask.T, axis=0)

        # Resize the slice to 256x256
        slc_pil = Image.fromarray(slc_img, mode="L").convert("RGBA").resize((256, 256))

        # Mask overlay
        overlay = Image.new("RGBA", slc_pil.size, (0, 0, 0, 0))  # Transparent image
        mask_resized = Image.fromarray(mask_img, mode="L").resize((256, 256), Image.NEAREST)

        # Add overlay based on the mask value
        for x in range(mask_resized.width):
            for y in range(mask_resized.height):
                if mask_resized.getpixel((x, y)) == 1:
                    overlay.putpixel((x, y), (255, 0, 0, 100))  # Red for anterior
                elif mask_resized.getpixel((x, y)) == 2:
                    overlay.putpixel((x, y), (0, 255, 0, 100))  # Green for posterior

        combined = Image.alpha_composite(slc_pil, overlay)

        # Position the combined image on the report
        pimg.paste(combined, box=(10 + i * 270, y_offset))

    return pimg



def anonymize_header(header):
    """
    Anonymizes the given DICOM header to comply with HIPAA guidelines.
    
    Arguments:
        header {pydicom.Dataset} -- the original DICOM header
    
    Returns:
        pydicom.Dataset -- the anonymized DICOM header
    """
    
    # List of tags to anonymize as DICOM tag tuples
    anonymization_tags = [
        (0x0010, 0x0010),  # PatientName
        (0x0010, 0x0020),  # PatientID
        (0x0010, 0x0030),  # PatientBirthDate
        (0x0010, 0x0040),  # PatientSex
        (0x0038, 0x0010),  # OtherPatientIDs
        (0x0038, 0x0011),  # OtherPatientNames
        (0x0008, 0x0050),  # AccessionNumber
        (0x0008, 0x0080),  # InstitutionName
        (0x0008, 0x0090),  # ReferringPhysicianName
        (0x0020, 0x0010),  # StudyID
        (0x0020, 0x000D),  # StudyInstanceUID
        (0x0020, 0x000E),  # SeriesInstanceUID
        (0x0008, 0x0018),  # SOPInstanceUID
        (0x0040, 0x0250),  # PerformedProcedureStepDescription
        (0x0040, 0x0251),  # CommentsOnThePerformedProcedureStep
        (0x0040, 0x0241),  # OperatorName (correct tag)
        (0x0008, 0x0070),  # Manufacturer
        (0x0050, 0x0001),  # SoftwareVersions
        (0x0050, 0x0010),  # StationName
    ]
    
    # Create a new Dataset instance to avoid modifying the original
    anonymized_header = pydicom.Dataset(header)
    
    # Anonymize sensitive data
    for tag in anonymization_tags:
        # Check if the tag exists in the header
        if tag in anonymized_header:
            data_element = anonymized_header.data_element(tag)
            if data_element is not None:
                data_element.value = ""  # Anonymize the value

    # Remove identification data related to DICOM series and instances
    anonymized_header.remove_private_tags()
    
    # Generate new UIDs to prevent any risk of re-identification
    anonymized_header.SOPInstanceUID = pydicom.uid.generate_uid()
    anonymized_header.SeriesInstanceUID = pydicom.uid.generate_uid()
    anonymized_header.StudyInstanceUID = pydicom.uid.generate_uid()
    
    # Optionally: anonymize dates and times, replacing with a valid value or removing them
    if 'StudyDate' in anonymized_header:
        anonymized_header.StudyDate = "19000101"  # Valid default date in DICOM
    if 'StudyTime' in anonymized_header:
        anonymized_header.StudyTime = "000000"  # Valid default time in DICOM
    
    return anonymized_header



def save_report_as_dcm(header, report, path):
    """Writes the supplied image as a DICOM Secondary Capture file"""
    
    # Anonymize the header before proceeding
    header = anonymize_header(header)

    # The rest of the code to save the DICOM report
    out = pydicom.Dataset(header)

    out.file_meta = pydicom.Dataset()
    out.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    out.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    out.file_meta.MediaStorageSOPClassUID = out.SOPClassUID

    out.SeriesInstanceUID = pydicom.uid.generate_uid()
    out.SOPInstanceUID = pydicom.uid.generate_uid()
    out.file_meta.MediaStorageSOPInstanceUID = out.SOPInstanceUID
    out.Modality = "OT"  # Other
    out.SeriesDescription = "HippoVolume.AI"

    out.Rows = report.height
    out.Columns = report.width

    out.ImageType = r"DERIVED\PRIMARY\AXIAL"  # We are deriving this image from patient data
    out.SamplesPerPixel = 3  # RGB image
    out.PhotometricInterpretation = "RGB"
    out.PlanarConfiguration = 0
    out.BitsAllocated = 8
    out.BitsStored = 8
    out.HighBit = 7
    out.PixelRepresentation = 0

    # Set time and date
    dt = datetime.date.today().strftime("%Y%m%d")
    tm = datetime.datetime.now().strftime("%H%M%S")
    out.StudyDate = dt
    out.StudyTime = tm
    out.SeriesDate = dt
    out.SeriesTime = tm

    out.ImagesInAcquisition = 1

    out.BurnedInAnnotation = "YES"

    out.PixelData = report.tobytes()

    # Correction: avoid using 'OperatorName' as a string
    # Make sure there is no 'OperatorName' field in the dataset
    operator_name_tag = (0x0040, 0x0241)  # The DICOM tag for 'OperatorName'
    if operator_name_tag in out:
        del out[operator_name_tag]  # Remove 'OperatorName' field if present

    # Fix the usage of 'write_like_original' (replace with 'enforce_file_format')
    pydicom.filewriter.dcmwrite(path, out, enforce_file_format=True)






def get_series_for_inference(study_dir):
    dicom_files = []
    for root, dirs, files in os.walk(study_dir):
        for file in files:
            if file.endswith(".dcm"):  # Vérifie si c'est un fichier DICOM
                dicom_files.append(pydicom.dcmread(os.path.join(root, file)))  # Charger le fichier DICOM
    return dicom_files

# Function get_predicted_volumes
def get_predicted_volumes(prediction):
    """
    Calculates the predicted volumes from the segmentation masks generated by the model.
    The volumes are calculated based on the predicted pixels in the masks.
    
    Arguments:
    prediction -- The numpy array containing the predicted segmentation masks (dimensions: [N, H, W])
    
    Returns a dictionary containing the volumes: anterior, posterior, and total.
    """
    # Convert the mask into a binary image
    anterior_mask = prediction == 1  # Anterior (1)
    posterior_mask = prediction == 2  # Posterior (2)

    # Calculate the volumes in mm³ (assuming each voxel is 1 mm³)
    anterior_volume = np.sum(anterior_mask)
    posterior_volume = np.sum(posterior_mask)
    total_volume = anterior_volume + posterior_volume

    return {
        'anterior': anterior_volume,
        'posterior': posterior_volume,
        'total': total_volume
    }




def os_command(command):
    # Comment this if running under Windows
    sp = subprocess.Popen(["/bin/bash", "-i", "-c", command])
    sp.communicate()

    # Uncomment this if running under Windows
    os.system(command)   




if __name__ == "__main__":
    import torch.nn.functional as F

    study_dir = "/content/nd320-c3-3d-imaging-starter/data/TestVolumes"
    print(f"Looking for series to run inference on in directory {study_dir}...")

    # Load DICOM files
    dicom_files = get_series_for_inference(study_dir)
    volume, header = load_dicom_volume_as_numpy_from_list(dicom_files)
    print(f"Found series of {volume.shape[0]} slices")

    # Create inference agent
    model_path = "/content/clean_dataset/result/2025-04-26_2151_Basic_unet/model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file '{model_path}' could not be found.")

    agent = UNetInferenceAgent(
        device="cuda" if torch.cuda.is_available() else "cpu",
        parameter_file_path=model_path
    )

    # Normalization
    volume = volume.astype(np.float32)
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

    # Resizing slice by slice
    slices = []
    for i in range(volume.shape[0]):
        slice_ = volume[i]  # [H, W]
        slice_tensor = torch.tensor(slice_, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        resized = F.interpolate(slice_tensor, size=(256, 256), mode='bilinear', align_corners=False)
        slices.append(resized.squeeze(0))  # [1, 256, 256]

    # Stack the volume
    volume_tensor_resized = torch.stack(slices)  # [N, 1, 256, 256]

    # ❗ Squeeze to remove the channel dimension of size 1
    volume_numpy_for_inference = volume_tensor_resized.squeeze(1).numpy()  # [N, 256, 256]

    # Inference
    prediction = agent.single_volume_inference(volume_numpy_for_inference)
    print("Prediction shape:", prediction.shape)

    # Calculate volumes
    inference_result = get_predicted_volumes(prediction)

    # Generate and save the report
    report_img = create_report(inference_result, header, volume, prediction)
    report_path = os.path.join(study_dir, "report.dcm")
    save_report_as_dcm(header, report_img, report_path)
    print(f"Report saved to {report_path}")

    # ✅ Send the report to the Orthanc server using storescu
    os_command(f"storescu 127.0.0.1 4242 {report_path}")

    # Clean up the folder
    time.sleep(2)
    #shutil.rmtree(study_dir, onerror=lambda f, p, e: print(f"Error deleting: {e[1]}"))

    # Summary
    print(f"Inference successful on {header['SOPInstanceUID'].value}, "
          f"out: {prediction.shape}, "
          f"anterior volume: {inference_result['anterior']}, "
          f"posterior volume: {inference_result['posterior']}, "
          f"total volume: {inference_result['total']}")
