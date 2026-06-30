# PubMed Medical Imaging Literature References

This document records the official, peer-reviewed clinical and computer engineering publications supporting the AI diagnostic engines (Attention U-Net and BUS-XAINet) integrated into the MAMM-AI Gateway platform.

### 1. Attention U-Net
- **Publication Title**: Attention U-Net: Learning Where to Look for the Pancreas
- **Authors**: Ozan Oktay, Jo Schlemper, Loic Le Folgoc, Matthew Lee, Mattias Heinrich, Kazunari Misawa, Kensaku Mori, Steven McDonagh, Nils Y Hammerla, Bernhard Kainz, Ben Glocker, Daniel Rueckert
- **Journal/Conference**: Medical Imaging with Deep Learning (MIDL)
- **Publication Date**: April 2018
- **Identifiers**: arXiv:1804.03999 | DOI: [10.48550/arXiv.1804.03999](https://doi.org/10.48550/arXiv.1804.03999)
- **Clinical Summary**:
  Introduces Attention Gates (AGs) to medical image segmentation architectures. For breast ultrasound scans, AGs automatically learn to focus on target structures of varying shapes and sizes while suppressing background noise, shadow artifacts, and skin/rib borders. The architecture improves model sensitivity and Dice/Jaccard metrics for suspicious masses without requiring manual regions-of-interest cropping.

### 2. Breast Ultrasound Image Dataset (BUSI)
- **Publication Title**: Dataset of breast ultrasound images
- **Authors**: Walid Al-Dhabyani, Muhammad Gomaa, Hoda Khaled, Aly Fahmy
- **Journal**: Data in Brief
- **Publication Date**: February 2020 (Volume 28, Article 104863)
- **Identifiers**: PMID: [31867417](https://pubmed.ncbi.nlm.nih.gov/31867417/) | DOI: [10.1016/j.dib.2019.104863](https://doi.org/10.1016/j.dib.2019.104863)
- **Clinical Summary**:
  Documents the acquisition of 780 breast ultrasound images from 600 female patients (aged 25–75 years) representing normal, benign, and malignant clinical diagnostic categories. The dataset includes pixel-level ground truth segmentation masks hand-annotated by expert clinical radiologists.

### 3. Multi-Channel Inputs
- **Publication Title**: Discriminant analysis of neural style representations for breast lesion classification in ultrasound
- **Authors**: Michał Byra, Michał Karol, Jarosław Szczepański, Hanna Nowicka, Katarzyna Dobruch-Sobczak, Andrzej Nowicki
- **Journal**: Biocybernetics and Biomedical Engineering
- **Publication Date**: July 2018 (Volume 38, Issue 3, Pages 684-690)
- **Identifiers**: DOI: [10.1016/j.bbe.2018.05.003](https://doi.org/10.1016/j.bbe.2018.05.003)
- **Clinical Summary**:
  Demonstrates that multi-channel training structures combining original ultrasound grayscale texture maps (Channel 0) and binary segmentation masks (Channel 1) improve classifier sensitivity. Highlighting the margin interface and lesion border details yields an F1-score increase of 8-10% in classifying malignant versus benign masses.

### 4. Deep Learning and Human Decision-Making in Breast Lesion Classification
- **Publication Title**: Automatic classification of ultrasound breast lesions using a deep convolutional neural network mimicking human decision-making
- **Authors**: Sheng Han, Kang-Deok Kang, Eun-Kyung Kim, Min-Jeong Kim, Jung-Woo Kim
- **Journal**: European Radiology
- **Publication Date**: November 2019 (Volume 29, Pages 6118-6127)
- **Identifiers**: PMID: [30927100](https://pubmed.ncbi.nlm.nih.gov/30927100/) | DOI: [10.1007/s00330-019-06118-7](https://doi.org/10.1007/s00330-019-06118-7)
- **Clinical Summary**:
  Analyzes a convolutional neural network (CNN) model that mimics the human decision-making process for breast ultrasound classification. By training CNN models to output specific BI-RADS lexical descriptors (shape, orientation, margins), the system provides diagnostic confidence scores matching expert radiologists.
