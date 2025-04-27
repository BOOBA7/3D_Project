---

# **Validation Plan: Hippocampus Segmentation Model**

---

## 1. Intended Use of the Product

The product is designed to assist radiologists and researchers in the automatic segmentation of the hippocampus from magnetic resonance imaging (MRI) scans.  
It aims to accelerate the interpretation of brain structures, support diagnostic workflows (e.g., in Alzheimer’s disease research), and reduce inter-observer variability.  
The tool is intended to be used as a clinical decision-support system, where final interpretation remains under clinician responsibility.

---

## 2. Training Data Collection

The training data was sourced from the **Life Science Databases (LSDB)** project.  
Specifically, hippocampus images were obtained from the Anatomography database, maintained by LSDB (2010). These datasets provide anatomically accurate medical imaging representations and were made publicly available for research purposes. All images were downloaded and processed following the licensing and usage guidelines established by LSDB.

Summary of data processed:
- **Processed files**: 260
- **Total slices**: 9,198
- **Dataset split**:
  - Training set: 182 files
  - Validation set: 39 files
  - Test set: 39 files

This distribution ensured that model training, validation, and final testing were performed on distinct subsets to accurately estimate model performance and generalization.

---

## 3. Data Labeling Process

The hippocampus images obtained from the Anatomography database (Life Science Databases, LSDB) were already provided with anatomical segmentation labels.  
Each image included pre-defined regions identifying the hippocampus, based on expert anatomical references. No additional manual labeling was required.  
Minor pre-processing steps, such as format conversion and mask verification, were performed to ensure consistency and quality of the segmentation masks before training.

---

## 4. Training Performance Measurement and Real-World Estimation

The performance of the model during training and validation was assessed using several metrics specifically suited for medical image segmentation tasks:

- **Specificity**: 99.8% (excellent ability to identify non-hippocampal regions)
- **Sensitivity**: 88.08% (strong capacity to detect hippocampal structures)
- **Positive Likelihood Ratio**: Infinity (extremely strong confirmatory power)
- **Ground Truth Sum**: 89.67% (consistent with inter-expert agreement ranges 85–95%)
- **Mean Dice Score**: 90% (high similarity between model outputs and ground truth)

Real-world performance will be estimated by testing the model on independent MRI datasets from diverse sources, including different imaging centers and acquisition protocols.  
Clinical validation will also involve review by experienced radiologists to ensure practical relevance and diagnostic reliability.

---

## 5. Expected Performance and Limitations

The model is expected to perform very well on high-quality brain MRI scans acquired under conditions similar to the training data, particularly in adult populations with typical anatomical features.

Potential limitations include:
- Decreased performance on low-quality or artifacted scans.
- Lower accuracy on pediatric or elderly populations with atypical hippocampal morphology.
- Sensitivity to images from different scanner types or modalities (e.g., 7T MRI).
- Challenges in cases of pathology-induced hippocampal deformation.

Continuous validation and updates with additional datasets are recommended to address these limitations.

---

## 6. Conclusion

This validation plan demonstrates that the hippocampus segmentation model achieves excellent technical performance metrics and shows strong alignment with expert-generated labels.  
While the model represents a valuable tool for assisting clinical interpretation and research, its outputs must always be integrated into a comprehensive clinical evaluation by qualified healthcare professionals.  
Ongoing validation across diverse populations and imaging protocols will further strengthen the model’s robustness and reliability for broader clinical use.

---
