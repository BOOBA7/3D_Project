"""
Preparing the dataset for hippocampus segmentation
In this notebook you will use the skills and methods that we have talked about during our EDA Lesson to prepare the hippocampus dataset using Python. Follow the Notebook, writing snippets of code where directed so using Task comments, similar to the one below, which expects you to put the proper imports in place. Write your code directly in the cell with TASK comment. Feel free to add cells as you see fit, but please make sure that code that performs that tasked activity sits in the same cell as the Task comment.


[ ]
# TASK: Import the following libraries that we will use: nibabel, matplotlib, numpy
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
It will help your understanding of the data a lot if you were able to use a tool that allows you to view NIFTI volumes, like 3D Slicer. I will refer to Slicer throughout this Notebook and will be pasting some images showing what your output might look like.

Loading NIFTI images using NiBabel
NiBabel is a python library for working with neuro-imaging formats (including NIFTI) that we have used in some of the exercises throughout the course. Our volumes and labels are in NIFTI format, so we will use nibabel to load and inspect them.

NiBabel documentation could be found here: https://nipy.org/nibabel/

Our dataset sits in two directories - images and labels. Each image is represented by a single file (we are fortunate to have our data converted to NIFTI) and has a corresponding label file which is named the same as the image file.

Note that our dataset is "dirty". There are a few images and labels that are not quite right. They should be quite obvious to notice, though. The dataset contains an equal amount of "correct" volumes and corresponding labels, and you don't need to alter values of any samples in order to get the clean dataset.

Since I am working on Google Colab, I will clone the Git repository to access the project's dataset.


[ ]
!rm -rf nd320-c3-3d-imaging-starter
!git clone https://github.com/udacity/nd320-c3-3d-imaging-starter.git


Cloning into 'nd320-c3-3d-imaging-starter'...
remote: Enumerating objects: 1291, done.
remote: Counting objects: 100% (56/56), done.
remote: Compressing objects: 100% (38/38), done.
remote: Total 1291 (delta 25), reused 18 (delta 18), pack-reused 1235 (from 1)
Receiving objects: 100% (1291/1291), 178.23 MiB | 15.55 MiB/s, done.
Resolving deltas: 100% (86/86), done.
Updating files: 100% (949/949), done.
Import the dataset using nibabel. Create two variables to store the image and label data.


[ ]
import os

base_path = '/content/nd320-c3-3d-imaging-starter/data/TrainingSet'
print("Subdirectories:", os.listdir(base_path))

print("\nImages:", os.listdir(os.path.join(base_path, "images"))[:5])
print("\nLabels:", os.listdir(os.path.join(base_path, "labels"))[:5])
Subdirectories: ['labels', 'images']

Images: ['hippocampus_145.nii.gz', 'hippocampus_053.nii.gz', 'hippocampus_351.nii.gz', 'hippocampus_228.nii.gz', 'hippocampus_352.nii.gz']

Labels: ['hippocampus_145.nii.gz', 'hippocampus_053.nii.gz', 'hippocampus_351.nii.gz', 'hippocampus_228.nii.gz', 'hippocampus_352.nii.gz']

[ ]
# List the files in the images and labels directories
image_dir = '/content/nd320-c3-3d-imaging-starter/data/TrainingSet/images'
label_dir = '/content/nd320-c3-3d-imaging-starter/data/TrainingSet/labels'

# List the files
image_files = [f for f in os.listdir(image_dir) if f.endswith('.nii.gz')]
label_files = [f for f in os.listdir(label_dir) if f.endswith('.nii.gz')]

# Check the number of images and labels
print(f"Number of images: {len(image_files)}")
print(f"Number of labels: {len(label_files)}")

# Verify that the number of images matches the number of labels
if len(image_files) == len(label_files) == 260:
    print("Correct number of images and labels (260 each).")
else:
    print("Mismatch detected between the number of images and labels.")


Number of images: 263
Number of labels: 262
Mismatch detected between the number of images and labels.
Create two variables to store the image and label data. Since this was already done in the previous code, I am simply following the provided instructions.


[ ]
# TASK: Load an image and a segmentation mask into variables called image and label
base_path = '/content/nd320-c3-3d-imaging-starter/data/TrainingSet'

# Example file paths
image_path = "/content/nd320-c3-3d-imaging-starter/data/TrainingSet/images/hippocampus_058.nii.gz"
label_path = "/content/nd320-c3-3d-imaging-starter/data/TrainingSet/labels/hippocampus_058.nii.gz"

# Load the image and label
image = nib.load(image_path)
label = nib.load(label_path)

# Convert to NumPy arrays
image_data = image.get_fdata()
label_data = label.get_fdata()

# Display the shape of the data
print(f"Image shape: {image_data.shape}")
print(f"Label shape: {label_data.shape}")
Image shape: (34, 53, 36)
Label shape: (34, 53, 36)
Visualize a random image and its corresponding label.


[ ]
import matplotlib.pyplot as plt

slice_index = 13  

plt.figure(figsize=(10, 5))

# Display the image
plt.subplot(1, 2, 1)
plt.imshow(image_data[:, :, slice_index], cmap="gray")
plt.title("Image")
plt.axis("off")

# Overlay the image and the mask
plt.subplot(1, 2, 2)
plt.imshow(image_data[:, :, slice_index], cmap="gray")
plt.imshow(label_data[:, :, slice_index], cmap="Reds", alpha=0.4)  # Transparent overlay
plt.title("Image + Mask")
plt.axis("off")

plt.tight_layout()
plt.show()




[ ]
# Nibabel can present your image data as a Numpy array by calling the method get_fdata()
# The array will contain a multi-dimensional Numpy array with numerical values representing voxel intensities.
# In our case, images and labels are 3-dimensional, so get_fdata will return a 3-dimensional array. You can verify this
# by accessing the .shape attribute. What are the dimensions of the input arrays?

# TASK: using matplotlib, visualize a few slices from the dataset, along with their labels.
# You can adjust plot sizes like so if you find them too small:
# plt.rcParams["figure.figsize"] = (10,10)

# TASK: Using matplotlib, visualize a few slices from the dataset along with their labels
# Visualize a few slices
plt.figure(figsize=(10,10))
plt.subplot(1, 2, 1)
plt.imshow(image_data[:, :, image_data.shape[2] // 4], cmap="gray")
plt.title("Image Slice")

plt.subplot(1, 2, 2)
plt.imshow(label_data[:, :, label_data.shape[2] // 4], cmap="jet")
plt.title("Label Slice")
plt.show()


Load volume into 3D Slicer to validate that your visualization is correct and get a feel for the shape of structures.Try to get a visualization like the one below (hint: while Slicer documentation is not particularly great, there are plenty of YouTube videos available! Just look it up on YouTube if you are not sure how to do something)

3D slicer


[ ]
# Stand out suggestion: use one of the simple Volume Rendering algorithms that we've
# implemented in one of our earlier lessons to visualize some of these volumes
# --- Volume Rendering avec marching cubes sur les labels ---
# Extraire la matrice affine pour obtenir les informations d'échelle
affine = image.affine
pixel_spacing = np.abs(affine[:3, :3].diagonal())  # Espacement entre les voxels

# %%
# Save the axial slice
print("Saving axial: ")
# Extract the axial slice (middle of the z-axis)
axial = image_data[image_data.shape[0] // 2, :, :]
print(axial.shape)
plt.imshow(axial, cmap="gray")
# Save with a full-range window
im = Image.fromarray((axial / np.max(axial) * 0xff).astype(np.uint8), mode="L")
im.save("axial.png")

# %%
# Save the **axial** slice
print("Saving axial:")
axial = image_data[:, :, image_data.shape[2] // 2]
plt.imshow(axial, cmap="gray")
im = Image.fromarray((axial / np.max(axial) * 0xff).astype(np.uint8), mode="L")
im.save("axial.png")

# Save the **sagittal** slice
print("Saving sagittal:")
sagittal = image_data[image_data.shape[0] // 2, :, :]
aspect = pixel_spacing[2] / pixel_spacing[1]
plt.imshow(sagittal, cmap="gray", aspect=aspect)
im = Image.fromarray((sagittal / np.max(sagittal) * 0xff).astype(np.uint8), mode="L")
im = im.resize((sagittal.shape[1], int(sagittal.shape[0] * aspect)))
im.save("sagittal.png")

# Save the **coronal** slice
print("Saving coronal:")
coronal = image_data[:, image_data.shape[1] // 2, :]
aspect = pixel_spacing[2] / pixel_spacing[0]
plt.imshow(coronal, cmap="gray", aspect=aspect)
im = Image.fromarray((coronal / np.max(coronal) * 0xff).astype(np.uint8), mode="L")
im = im.resize((coronal.shape[1], int(coronal.shape[0] * aspect)))
im.save("coronal.png")




[ ]
plt.imshow(axial, cmap="gray")


[ ]
plt.imshow(sagittal, cmap="gray", aspect=aspect)


[ ]
plt.imshow(coronal, cmap="gray", aspect=aspect)

3D visualization of the hippocampus.


[ ]
import plotly.graph_objects as go
import numpy as np

# Seulement les voxels où label > 0 (les structures annotées)
x, y, z = np.where(label_data > 0)

# Valeurs pour la couleur (par exemple 1 ou 2 selon la classe)
values = label_data[label_data > 0]

# Création du graphique 3D
fig = go.Figure(data=go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=3,
        color=values,       # couleurs selon la valeur du label
        colorscale='Viridis',
        opacity=0.6
    )
))

fig.update_layout(
    scene=dict(
        xaxis_title='Sagittal (x)',
        yaxis_title='Coronal (y)',
        zaxis_title='Axial (z)',
    ),
    title='Visualisation 3D du masque de segmentation (hippocampe)',
    margin=dict(l=0, r=0, t=30, b=0)
)

fig.show()



Looking at single image data
In this section we will look closer at the NIFTI representation of our volumes. In order to measure the physical volume of hippocampi, we need to understand the relationship between the sizes of our voxels and the physical world.


[ ]
# Nibabel supports many imaging formats, NIFTI being just one of them. I told you that our images
# are in NIFTI, but you should confirm if this is indeed the format that we are dealing with
# TASK: using .header_class attribute - what is the format of our images?

image.header_class



[ ]
header = image.header
print(header)
<class 'nibabel.nifti1.Nifti1Header'> object, endian='<'
sizeof_hdr      : 348
data_type       : np.bytes_(b'')
db_name         : np.bytes_(b'')
extents         : 0
session_error   : 0
regular         : np.bytes_(b'r')
dim_info        : 0
dim             : [ 3 34 53 36  1  1  1  1]
intent_p1       : 0.0
intent_p2       : 0.0
intent_p3       : 0.0
intent_code     : none
datatype        : float32
bitpix          : 32
slice_start     : 0
pixdim          : [1. 1. 1. 1. 1. 0. 0. 0.]
vox_offset      : 0.0
scl_slope       : nan
scl_inter       : nan
slice_end       : 0
slice_code      : unknown
xyzt_units      : 10
cal_max         : 0.0
cal_min         : 0.0
slice_duration  : 0.0
toffset         : 0.0
glmax           : 0
glmin           : 0
descrip         : np.bytes_(b'5.0.10')
aux_file        : np.bytes_(b'none')
qform_code      : scanner
sform_code      : scanner
quatern_b       : 0.0
quatern_c       : 0.0
quatern_d       : 0.0
qoffset_x       : 1.0
qoffset_y       : 1.0
qoffset_z       : 1.0
srow_x          : [1. 0. 0. 1.]
srow_y          : [0. 1. 0. 1.]
srow_z          : [0. 0. 1. 1.]
intent_name     : np.bytes_(b'')
magic           : np.bytes_(b'n+1')
As we can see in the NIfTI header, there is no metadata available, but it contains all the details regarding pixel and voxel information, as well as the dimensionality of the NIfTI file format.


[ ]
image.affine
array([[1., 0., 0., 1.],
       [0., 1., 0., 1.],
       [0., 0., 1., 1.],
       [0., 0., 0., 1.]])
Further down we will be inspecting .header attribute that provides access to NIFTI metadata. You can use this resource as a reference for various fields: https://brainder.org/2012/09/23/the-nifti-file-format/

Below, as requested, I will implement some code to extract information from the NIfTI file, such as pixel data, spacing, and dimensions.


[ ]
# Exemple
print("Shape of image:", image.shape)

Shape of image: (34, 53, 36)

[ ]
# TASK: How many bits per pixel are used?
print(f"Bits per pixel: {image.header['bitpix']}")
Bits per pixel: 32

[ ]
# TASK: What are the units of measurement?
print(f"Units of measurement: {image.header['xyzt_units']}")
Units of measurement: 10

[ ]
# TASK: Do we have a regular grid? What are grid spacings?
print(f"Grid spacings: {image.header['pixdim']}")
Grid spacings: [1. 1. 1. 1. 1. 0. 0. 0.]

[ ]
# TASK: What dimensions represent axial, sagittal, and coronal slices? How do you know?
# Les dimensions de l'image
print(f"Dimensions of the image: {image_data.shape}")
# En général, la 1ère dimension est sagittale, la 2ème est coronal, et la 3ème est axiale
print("This should be verified with the scanner's specifications")
# The dimensions of the image
print(f"Dimensions of the image: {image_data.shape}")

# Mapping between dimensions and anatomical planes
print(f"Sagittal slices (left-right): {image_data.shape[0]}")  # X axis
print(f"Coronal slices (front-back): {image_data.shape[1]}")   # Y axis
print(f"Axial slices (top-bottom): {image_data.shape[2]}")     # Z axis

# Important note
print("Note: This mapping assumes that the orientation was not altered during the DICOM to NIfTI conversion.")
print("It is recommended to check the affine matrix to be 100% sure about the spatial orientation.")
Dimensions of the image: (34, 53, 36)
This should be verified with the scanner's specifications
Dimensions of the image: (34, 53, 36)
Sagittal slices (left-right): 34
Coronal slices (front-back): 53
Axial slices (top-bottom): 36
Note: This mapping assumes that the orientation was not altered during the DICOM to NIfTI conversion.
It is recommended to check the affine matrix to be 100% sure about the spatial orientation.

[ ]
# By now you should have enough information to decide what are dimensions of a single voxel
# TASK: Compute the volume (in mm³) of a hippocampus using one of the labels you've loaded.
# You should get a number between ~2200 and ~4500

# TASK: Compute the volume (in mm³) of a hippocampus using one of the labels
# Calculer le volume (en mm³) de l'hippocampe
voxel_volume = np.prod(image.header['pixdim'][1:4])  # produit des dimensions des voxels
hippocampus_volume = np.sum(label_data == 1) * voxel_volume  # Volume de la classe 1 (hippocampe)
print(f"Hippocampus volume: {hippocampus_volume} mm³")

Hippocampus volume: 1334.0 mm³
Plotting some charts
The normal volume of the human hippocampus varies depending on several factors, such as age, sex, and individual health conditions. However, here are approximate averages:

For an average adult:

Combined volume of both hippocampi: about 3000 to 3500 mm³.

Volume of each hippocampus (right or left): about 1500 to 1750 mm³ for each side.

The volumes may decrease with age, especially in older individuals or those with neurodegenerative diseases such as Alzheimer's disease. A reduction in hippocampal volume is often an indicator of cognitive decline.


[ ]
# Folder paths
base_path = "/content/nd320-c3-3d-imaging-starter/data/TrainingSet"
image_dir = os.path.join(base_path, "images")
label_dir = os.path.join(base_path, "labels")

# Clean up files (remove .DS_Store)
image_files = sorted([f for f in os.listdir(image_dir)
                     if f.endswith('.nii.gz') and not f.startswith('.')])
label_files = sorted([f for f in os.listdir(label_dir)
                     if f.endswith('.nii.gz') and not f.startswith('.')])

print(f"Valid images: {len(image_files)}, Valid labels: {len(label_files)}")

# Extract numeric IDs
def extract_id(filename):
    """Extracts the numeric ID from the filename"""
    return filename.split('_')[-1].split('.')[0]  # Takes the number after hippocampus_

image_ids = {extract_id(f): f for f in image_files}
label_ids = {extract_id(f): f for f in label_files}

# Matching pairs
common_ids = set(image_ids.keys()) & set(label_ids.keys())
valid_pairs = [(image_ids[id], label_ids[id]) for id in common_ids]

print(f"\nValid pairs found: {len(valid_pairs)}")
print("Example of a pair:", valid_pairs[0] if valid_pairs else "None")

# Volume calculation (only for valid pairs)
volumes = []
for img_file, lbl_file in valid_pairs:
    img = nib.load(os.path.join(image_dir, img_file))
    lbl = nib.load(os.path.join(label_dir, lbl_file))

    voxel_volume = np.prod(img.header['pixdim'][1:4])
    hippocampus_vol = np.sum(lbl.get_fdata() == 1) * voxel_volume
    volumes.append(hippocampus_vol)

# Visualization
plt.figure(figsize=(10, 6))
plt.hist(volumes, bins=30, color='skyblue', edgecolor='black')
plt.title(f"Volume Distribution ({len(volumes)} valid subjects)")
plt.xlabel("Volume (mm³)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

In the representation of the distribution of volumes in our data, we observe some anomalies that may be due to errors in the image data. Below, we will proceed to clean our data by standardizing the hippocampus volumes using a minimum and maximum range, ensuring a more accurate and realistic representation.


[ ]
# Define the reference range
REFERENCE_RANGE = (1000, 6500)  # mm³

# Convert to numpy array for vectorized analysis
volumes_array = np.array(volumes)

# 1. Identify out-of-range values
out_of_range_mask = (volumes_array < REFERENCE_RANGE[0]) | (volumes_array > REFERENCE_RANGE[1])
out_of_range_volumes = volumes_array[out_of_range_mask]
out_of_range_indices = np.where(out_of_range_mask)[0]

# 2. Quantitative analysis
print(f"Range compliance analysis for {REFERENCE_RANGE} mm³:")
print(f"- Volumes analyzed: {len(volumes_array)}")
print(f"- Compliant volumes: {len(volumes_array) - len(out_of_range_volumes)}")
print(f"- Out-of-range volumes: {len(out_of_range_volumes)}")
print(f"- Percentage out of range: {len(out_of_range_volumes)/len(volumes_array):.1%}")

# 3. Details of non-compliant values
if len(out_of_range_volumes) > 0:
    print("\nDetails of out-of-range values:")
    print(f"- Abnormal minimum: {np.min(out_of_range_volumes):.2f} mm³")
    print(f"- Abnormal maximum: {np.max(out_of_range_volumes):.2f} mm³")
    print(f"- Mean of out-of-range values: {np.mean(out_of_range_volumes):.2f} mm³")

    # Associating with files
    print("\nExamples of affected files:")
    for idx in out_of_range_indices[:3]:  # Display the first 3
        img_file, lbl_file = valid_pairs[idx]
        print(f"  {img_file} : {volumes_array[idx]:.2f} mm³")

Range compliance analysis for (1000, 6500) mm³:
- Volumes analyzed: 262
- Compliant volumes: 260
- Out-of-range volumes: 2
- Percentage out of range: 0.8%

Details of out-of-range values:
- Abnormal minimum: 853.69 mm³
- Abnormal maximum: 95716.21 mm³
- Mean of out-of-range values: 48284.95 mm³

Examples of affected files:
  hippocampus_010.nii.gz : 853.69 mm³
  hippocampus_281.nii.gz : 95716.21 mm³


Do you see any outliers? Why do you think it's so (might be not immediately obvious, but it's always a good idea to inspect) outliers closer. If you haven't found the images that do not belong, the histogram may help you.

In the real world we would have precise information about the ages and conditions of our patients, and understanding how our dataset measures against population norm would be the integral part of clinical validation that we talked about in last lesson. Unfortunately, we do not have this information about this dataset, so we can only guess why it measures the way it is. If you would like to explore further, you can use the calculator from HippoFit project to see how our dataset compares against different population slices

Did you notice anything odd about the label files? We hope you did! The mask seems to have two classes, labeled with values 1 and 2 respectively. If you visualized sagittal or axial views, you might have gotten a good guess of what those are. Class 1 is the anterior segment of the hippocampus and class 2 is the posterior one.

For the purpose of volume calculation we do not care about the distinction, however we will still train our network to differentiate between these two classes and the background


[ ]
import shutil

# Folder paths
base_path = "/content/nd320-c3-3d-imaging-starter/data/TrainingSet"
clean_base = "/content/clean_dataset"  # Output folder

# Create output folders
os.makedirs(os.path.join(clean_base, "images"), exist_ok=True)
os.makedirs(os.path.join(clean_base, "labels"), exist_ok=True)

# 1. Filter valid pairs (outliers excluded)
clean_pairs = [pair for i, pair in enumerate(valid_pairs) if not out_of_range_mask[i]]

# 2. Copy files
for img_file, lbl_file in clean_pairs:
    # Copy images
    shutil.copy(
        os.path.join(base_path, "images", img_file),
        os.path.join(clean_base, "images", img_file)
    )
    # Copy labels
    shutil.copy(
        os.path.join(base_path, "labels", lbl_file),
        os.path.join(clean_base, "labels", lbl_file)
    )

# 3. Final check
clean_images = os.listdir(os.path.join(clean_base, "images"))
clean_labels = os.listdir(os.path.join(clean_base, "labels"))

print(f"\nCleaned dataset created in {clean_base}")
print(f"- Images saved: {len(clean_images)}")
print(f"- Labels saved: {len(clean_labels)}")

# 4. Save metadata (corrected version)
metadata = []
for img_file, lbl_file in clean_pairs:
    # Find the index of the pair in valid_pairs
    pair_index = valid_pairs.index((img_file, lbl_file))
    metadata.append({
        "image": img_file,
        "label": lbl_file,
        "volume": volumes_array[pair_index]  # Simplified and corrected version
    })

# Save the CSV
pd.DataFrame(metadata).to_csv(os.path.join(clean_base, "metadata.csv"), index=False)
print("metadata.csv file successfully created")

Cleaned dataset created in /content/clean_dataset
- Images saved: 260
- Labels saved: 260
metadata.csv file successfully created

[ ]
# Load and display a sample of the metadata
df = pd.read_csv(os.path.join(clean_base, "metadata.csv"))
print("\nMetadata preview:")
print(df.head())

# Check the volumes
print("\nVolume check:")
print(f"Minimum volume: {df['volume'].min():.2f} mm³")
print(f"Maximum volume: {df['volume'].max():.2f} mm³")


Metadata preview:
                    image                   label  volume
0  hippocampus_394.nii.gz  hippocampus_394.nii.gz  1987.0
1  hippocampus_354.nii.gz  hippocampus_354.nii.gz  1529.0
2  hippocampus_096.nii.gz  hippocampus_096.nii.gz  1890.0
3  hippocampus_074.nii.gz  hippocampus_074.nii.gz  1438.0
4  hippocampus_097.nii.gz  hippocampus_097.nii.gz  1354.0

Volume check:
Minimum volume: 1054.00 mm³
Maximum volume: 2593.00 mm³

[ ]
# List of files in the images and labels folders
label_dir = '/content/clean_dataset/images'
image_dir = '/content/clean_dataset/labels'
output_dir = '/content/clean_dataset'  # You can also use /content to stay in the correct directory
os.makedirs(output_dir, exist_ok=True)

# List files
image_files = [f for f in os.listdir(image_dir) if f.endswith('.nii.gz')]
label_files = [f for f in os.listdir(label_dir) if f.endswith('.nii.gz')]

# Check the number of images and labels
print(f"Number of images: {len(image_files)}")
print(f"Number of labels: {len(label_files)}")

# Verify that the number of images and labels match
if len(image_files) == len(label_files) == 260:
    print("Correct number of images and labels (260 each).")
else:
    print("There is a mismatch in the number of images or labels.")

Number of images: 260
Number of labels: 260
Correct number of images and labels (260 each).
Final remarks
Congratulations! You have finished Section 1.

In this section you have inspected a dataset of MRI scans and related segmentations, represented as NIFTI files. We have visualized some slices, and understood the layout of the data. We have inspected file headers to understand what how the image dimensions relate to the physical world and we have understood how to measure our volume. We have then inspected dataset for outliers, and have created a clean set that is ready for consumption by our ML algorithm.

In the next section you will create training and testing pipelines for a UNet-based machine learning model, run and monitor the execution, and will produce test metrics. This will arm you with all you need to use the model in the clinical context and reason about its performance!


[ ]
!pip install medpy
!pip install utils
Collecting medpy
  Downloading medpy-0.5.2.tar.gz (156 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 156.3/156.3 kB 2.4 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
Requirement already satisfied: scipy>=1.10 in /usr/local/lib/python3.11/dist-packages (from medpy) (1.14.1)
Requirement already satisfied: numpy>=1.24 in /usr/local/lib/python3.11/dist-packages (from medpy) (2.0.2)
Collecting SimpleITK>=2.1 (from medpy)
  Downloading SimpleITK-2.4.1-cp311-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.9 kB)
Downloading SimpleITK-2.4.1-cp311-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (52.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 52.3/52.3 MB 20.2 MB/s eta 0:00:00
Building wheels for collected packages: medpy
  Building wheel for medpy (setup.py) ... done
  Created wheel for medpy: filename=MedPy-0.5.2-py3-none-any.whl size=224710 sha256=6c0d110fde1029bc9c540cb55efde2a369bcd8b3abb66fa1affa9027462c0375
  Stored in directory: /root/.cache/pip/wheels/d4/33/ed/aaac5a347fb8d41679ca515b8f5c49dfdf49be15bdbb9a905d
Successfully built medpy
Installing collected packages: SimpleITK, medpy
Successfully installed SimpleITK-2.4.1 medpy-0.5.2
Collecting utils
  Downloading utils-1.0.2.tar.gz (13 kB)
  Preparing metadata (setup.py) ... done
Building wheels for collected packages: utils
  Building wheel for utils (setup.py) ... done
  Created wheel for utils: filename=utils-1.0.2-py2.py3-none-any.whl size=13906 sha256=5b8026ee1f657516f45e05577eb6c8504fd331e9316afb7d110c454d5dfe9102
  Stored in directory: /root/.cache/pip/wheels/15/0c/b3/674aea8c5d91c642c817d4d630bd58faa316724b136844094d
Successfully built utils
Installing collected packages: utils
Successfully installed utils-1.0.2
Installation of the required libraries and dependencies.

This module defines a function to load and preprocess the hippocampus dataset, making it ready for model training.

LoadHippocampusData function:

This function loads the hippocampus dataset, which consists of 3D medical image volumes and their corresponding labels (segmentation masks).
It takes as input the root directory containing the dataset, as well as the desired output shape for the images.
The dataset is loaded from two directories: images for the image volumes and labels for the corresponding segmentation labels.
The function iterates over all image files, loading both the image and its label using the MedPy load function.
The images are normalized to the [0, 1] range by subtracting the minimum value and dividing by the range (max - min).
Both the images and labels are reshaped to a consistent size using the med_reshape function, ensuring they are compatible with the CNN input requirements.
The reshaped data is stored in a dictionary containing the image, segmentation mask, and the filename, which is then appended to a list.
med_reshape function:

This helper function reshapes a 3D image to a new size by padding it with zeros, leaving the original content in the top-left corner.
The new shape is specified as a 3-tuple, and the function ensures that no content is discarded during reshaping, with excess space padded with zeros.
Output:

The function returns a NumPy array of dictionaries, each containing the reshaped image and label data, making the dataset ready for model training.
The total number of processed slices and files is printed as a summary.
The dataset is assumed to fit into memory (about 300MB), so it is fully loaded into RAM for fast access during training.

NOTE: The dataset loading process can be optimized further for larger datasets that do not fit into memory by using techniques like memory-mapped files.


[ ]
"""
Module loads the hippocampus dataset into RAM
"""
import os
from os import listdir
from os.path import isfile, join

import numpy as np
from medpy.io import load

…    # TASK: write your original image into the reshaped image
    x_max = min(image.shape[0], new_shape[0])
    y_max = min(image.shape[1], new_shape[1])
    z_max = min(image.shape[2], new_shape[2])

    reshaped_image[:x_max, :y_max, :z_max] = image[:x_max, :y_max, :z_max]

    return reshaped_image


This module defines a custom PyTorch dataset class SlicesDataset for loading 3D medical image data. The dataset is designed to represent 2D slices of a 3D volume, which can be processed individually during training or inference. The dataset can be consumed by the PyTorch DataLoader for batching and shuffling.

SlicesDataset:

This class is a subclass of torch.utils.data.Dataset and represents an indexable dataset that can be used with the PyTorch DataLoader class.
The dataset is initialized with a list of dictionaries, where each dictionary contains 3D image and segmentation data. The class processes these 3D volumes into individual 2D slices.
Each slice of the image and segmentation data is stored in the slices list, which holds tuples of indices corresponding to the volume and slice number.
__getitem__ method:

Retrieves a specific slice of the 3D volume by index (idx), returning a dictionary containing two 3D tensors: one for the image ("image") and one for the segmentation ("seg").
The shape of each tensor is [1, H, W], where H and W are the height and width of the 2D slice.
You can implement caching or data augmentation strategies within this method if needed.
__len__ method:

Returns the total number of 2D slices available in the dataset, which is used by the DataLoader to determine the size of the dataset.
This dataset is useful when working with large 3D medical image volumes and allows for efficient batch processing of individual slices during training or inference.


[ ]
"""
Module for Pytorch dataset representations
"""

import torch
from torch.utils.data import Dataset

class SlicesDataset(Dataset):
    """
    This class represents an indexable Torch dataset
    which could be consumed by the PyTorch DataLoader class
    """
    def __init__(self, data):
        self.data = data

        self.slices = []

        for i, d in enumerate(data):
            for j in range(d["image"].shape[0]):
                self.slices.append((i, j))

    def __getitem__(self, idx):
        """
        This method is called by PyTorch DataLoader class to return a sample with id idx

        Arguments:
            idx {int} -- id of sample

        Returns:
            Dictionary of 2 Torch Tensors of dimensions [1, W, H]
        """
        slc = self.slices[idx]
        sample = dict()
        sample["id"] = idx

        # You could implement caching strategy here if dataset is too large to fit
        # in memory entirely
        # Also this would be the place to call transforms if data augmentation is used

        # TASK: Create two new keys in the "sample" dictionary, named "image" and "seg"
        # The values are 3D Torch Tensors with image and label data respectively.
        # First dimension is size 1, and last two hold the voxel data from the respective
        # slices. Write code that stores the 2D slice data in the last 2 dimensions of the 3D Tensors.
        # Your tensor needs to be of shape [1, patch_size, patch_size]
        # Don't forget that you need to put a Torch Tensor into your dictionary element's value
        # Hint: your 3D data sits in self.data variable, the id of the 3D volume from data array
        # and the slice number are in the slc variable.
        # Hint2: You can use None notation like so: arr[None, :] to add size-1
        # dimension to a Numpy array

        i, j = slc
        image_slice = self.data[i]["image"][j]  # shape: (H, W)
        label_slice = self.data[i]["seg"][j]    # shape: (H, W)

        sample["image"] = torch.tensor(image_slice[None, :, :], dtype=torch.float32)
        sample["seg"] = torch.tensor(label_slice[None, :, :], dtype=torch.long)

        return sample

    def __len__(self):
        """
        This method is called by PyTorch DataLoader class to return number of samples in the dataset

        Returns:
            int
        """
        return len(self.slices)

This module defines a recursive implementation of the UNet architecture used for image segmentation. It includes two main classes:

UNet:

The main UNet class, which constructs the network by stacking UnetSkipConnectionBlock blocks.
The number of downsampling steps (num_downs) determines the depth of the network, and the network progressively reduces the spatial dimensions before expanding back to the original size.
The final layer produces the segmentation output for the specified number of classes.
UnetSkipConnectionBlock:

Represents each block of the UNet architecture, handling the downsampling (contracting) and upsampling (expanding) steps, with skip connections between the encoder and decoder.
Supports both innermost and outermost blocks, as well as optional dropout for regularization.
Includes methods for contracting (downsampling), expanding (upsampling), and cropping features during the forward pass.
Key features:

Utilizes instance normalization and LeakyReLU activation functions.
Supports dynamic patch size, dropout, and handling of the outermost and innermost layers of the network.
Uses skip connections to preserve fine-grained spatial information during upsampling.

[ ]
# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Defines the Unet.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1 at the bottleneck

# recursive implementation of Unet
import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self, num_classes=3, in_channels=1, initial_filter_size=64, kernel_size=3, num_downs=4, norm_layer=nn.InstanceNorm2d):
        super(UNet, self).__init__()

        unet_block = UnetSkipConnectionBlock(in_channels=initial_filter_size * 2 ** (num_downs-1), out_channels=initial_filter_size * 2 ** num_downs,
                                             num_classes=num_classes, kernel_size=kernel_size, norm_layer=norm_layer, innermost=True)
        for i in range(1, num_downs):
            unet_block = UnetSkipConnectionBlock(in_channels=initial_filter_size * 2 ** (num_downs-(i+1)),
                                                 out_channels=initial_filter_size * 2 ** (num_downs-i),
                                                 num_classes=num_classes, kernel_size=kernel_size, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(in_channels=in_channels, out_channels=initial_filter_size,
                                             num_classes=num_classes, kernel_size=kernel_size, submodule=unet_block, norm_layer=norm_layer,
                                             outermost=True)

        self.model = unet_block

    def forward(self, x):
        return self.model(x)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, num_classes=1, kernel_size=3,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        pool = nn.MaxPool2d(2, stride=2)
        conv1 = self.contract(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, norm_layer=norm_layer)
        conv2 = self.contract(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, norm_layer=norm_layer)

        conv3 = self.expand(in_channels=out_channels*2, out_channels=out_channels, kernel_size=kernel_size)
        conv4 = self.expand(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)

        if outermost:
            final = nn.Conv2d(out_channels, num_classes, kernel_size=1)
            down = [conv1, conv2]
            up = [conv3, conv4, final]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(in_channels*2, in_channels,
                                        kernel_size=2, stride=2)
            model = [pool, conv1, conv2, upconv]
        else:
            upconv = nn.ConvTranspose2d(in_channels*2, in_channels, kernel_size=2, stride=2)
            down = [pool, conv1, conv2]
            up = [conv3, conv4, upconv]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, norm_layer=nn.InstanceNorm2d):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            norm_layer(out_channels),
            nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        return layer

    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            crop = self.center_crop(self.model(x), x.size()[2], x.size()[3])
            return torch.cat([x, crop], 1)


This module defines the UNetInferenceAgent class, which is responsible for running inference on 3D medical volumes using a trained UNet model. The class handles loading the model, processing input volumes, and generating predictions. It supports running inference on both padded and conformant volumes.

Key functionalities:

Initializes the model and loads parameters from a specified file path.
Runs inference on a single volume with a specified patch size.
Handles 3D volume processing by splitting it into 2D slices and performing predictions for each slice.
Supports both padded and non-padded inference.

[ ]
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
        # Pad the volume to match the patch size
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
                # Extraire la slice [Y,Z] et la convertir en un tenseur de forme [1, 1, Y, Z]
                slice_2d = volume[i, :, :]
                input_tensor = torch.tensor(slice_2d[None, None, :, :], dtype=torch.float32).to(self.device)

                # Prédiction
                output = self.model(input_tensor)

                # Convertir en classe prédite (argmax sur la dimension des canaux)
                pred_slice = torch.argmax(output, dim=1).cpu().numpy()[0]

                slices.append(pred_slice)

        # Empiler toutes les slices pour reconstruire le volume 3D
        return np.stack(slices, axis=0)

This module represents a UNet experiment and contains a class that handles the experiment lifecycle.

The class follows a typical UNet experiment workflow for segmentation tasks, and the basic lifecycle is as follows:

run(): This method runs for each epoch, calling train() and validate() for training and validation respectively.

train(): A single epoch training on the training dataset.

validate(): Evaluation on the validation dataset with additional metrics like loss, sensitivity, specificity, ground truth sum, and likelihood ratio.

test(): Inference on the test set.

Additional evaluation methods like sensitivity, specificity, ground truth sum, and likelihood ratio have been implemented to enrich the validation process. These metrics provide deeper insights into model performance beyond the basic loss, helping in more comprehensive evaluation during the validation phase.


[ ]
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

#rom data_prep.SlicesDataset import SlicesDataset
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

This module contains functions for calculating common similarity metrics between two 3D volumes: Dice Similarity Coefficient (Dice3d) and Jaccard Similarity Coefficient (Jaccard3d). These functions are designed to compare binary 3D masks, treating 0 as background and any non-zero value as part of the structure.

Dice3d(a, b): Computes the Dice Similarity Coefficient, which is a measure of the overlap between two binary 3D volumes. A value of 1.0 indicates perfect overlap, while a value of 0 indicates no overlap. It is often used in medical imaging for evaluating segmentation results.

Jaccard3d(a, b): Computes the Jaccard Similarity Coefficient, which is another metric to measure the similarity between two binary 3D volumes. A value of 1.0 indicates perfect similarity (complete overlap), and a value of 0 indicates no similarity.

Both functions handle inputs of 3D arrays (volumes) and ensure they are of the same shape and dimensionality before proceeding. If both volumes are empty (no structures), both functions return a similarity score of 1.0, considering it a perfect match.


[ ]
"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np

def Dice3d(a, b):
    """
    This will compute the Dice Similarity coefficient for two 3-dimensional volumes.
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data.

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # Convert to binary mask: 0 = background, 1 = structure
    a_bin = (a > 0).astype(np.uint8)
    b_bin = (b > 0).astype(np.uint8)

    intersection = np.sum(a_bin * b_bin)
    sum_volumes = np.sum(a_bin) + np.sum(b_bin)

    if sum_volumes == 0:
        return 1.0  # If both are empty, consider perfect match

    dice_score = 2.0 * intersection / sum_volumes
    return dice_score


def Jaccard3d(a, b):
    """
    This will compute the Jaccard Similarity coefficient for two 3-dimensional volumes.
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data.

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    a_bin = (a > 0).astype(np.uint8)
    b_bin = (b > 0).astype(np.uint8)

    intersection = np.sum(a_bin * b_bin)
    union = np.sum((a_bin + b_bin) > 0)

    if union == 0:
        return 1.0  # If both are empty, perfect match

    jaccard_score = intersection / union
    return jaccard_score


This module provides functions for evaluating 3D segmentation performance using additional metrics beyond Dice and Jaccard:

Sensitivity3d: Computes the Sensitivity (True Positive Rate) of the segmentation.

Specificity3d: Computes the Specificity (True Negative Rate) of the segmentation.

LikelihoodRatio3d: Computes the Likelihood Ratio, combining sensitivity and specificity.

full_additional_report: Generates a report with sensitivity, specificity, likelihood ratio, and ground truth sum.

save_report_as_json: Saves the evaluation report as a JSON file.

These metrics provide a more comprehensive assessment of the model's segmentation performance.


[ ]
"""
Module for additional 3D evaluation metrics
"""
import numpy as np
import json
import os

def Sensitivity3d(y_pred, y_true):
    y_pred_bin = (y_pred > 0).astype(np.uint8)
    y_true_bin = (y_true > 0).astype(np.uint8)

TensorBoard Logging Function

This function logs various metrics and images to TensorBoard during the training or validation phases of a model. It records:

Loss: The loss value for the current step/epoch.

Images: Input images, ground truth masks, and predicted masks.

Additional Metrics: Optional metrics like Sensitivity, Specificity, Likelihood Ratio, and Ground Truth Sum.

Key Steps:

Ensures the input tensors have the correct shape for logging.

Logs images (input, ground truth, prediction) at each step.

Logs additional metrics if provided, allowing for more detailed tracking of model performance.

This function helps visualize model performance and track training progress in TensorBoard.


"""

import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

def log_to_tensorboard(writer, loss, data, target, prediction_softmax, prediction, counter,
                       phase='train', sensitivity=None, specificity=None,
                       likelihood_ratio=None, ground_truth_sum=None):
    """
    Logs training or validation progress to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        loss: scalar loss
        data: input images
        target: ground truth masks
        prediction_softmax: softmax output from the model
        prediction: predicted masks
        counter: step or epoch number
        phase: 'train' or 'val'
        sensitivity, specificity, likelihood_ratio, ground_truth_sum: optional metrics
    """
    writer.add_scalar(f'Loss/{phase}', loss, counter)

    def ensure_nchw(tensor):
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor

    data = ensure_nchw(data)
    target = ensure_nchw(target)
    prediction_softmax = ensure_nchw(prediction_softmax[:, 1, :, :])  # Assuming class 1 is the region of interest

    writer.add_images(f'{phase}/Image', data, counter)
    writer.add_images(f'{phase}/GT', target, counter)
    writer.add_images(f'{phase}/Prediction', prediction_softmax, counter)

    # Log additional metrics if provided
    if sensitivity is not None:
        writer.add_scalar(f'{phase}/Sensitivity', sensitivity, counter)
    if specificity is not None:
        writer.add_scalar(f'{phase}/Specificity', specificity, counter)
    if likelihood_ratio is not None:
        writer.add_scalar(f'{phase}/Likelihood_Ratio', likelihood_ratio, counter)
    if ground_truth_sum is not None:
        writer.add_scalar(f'{phase}/GT_Sum', ground_truth_sum, counter)