# Image Classification using Transfer Learning (VGG16)

## Course Information

- **Course Title:** Machine Learning Lab  
- **Course Code:** CSE 432  
- **Submission Date:** 07 July 2025  

## Submitted by

- **Name:** Kayes Mahmood  
- **ID:** 2215151030  
- **Batch:** 51  
- **Section:** 7A2  

## Submitted to

- **Mrinmoy Biswas Akash**  
  Lecturer & Course Coordinator  
  Department of CSE  
  University of Information Technology and Sciences  

- **Md. Yousuf Ali**  
  Lecturer  
  Department of CSE  
  University of Information Technology and Sciences  

## GitHub Repository

[CashMahmood/Image-Classification-using-Transfer-Learning-Model-in-Machine-Learning](https://github.com/CashMahmood/Image-Classification-using-Transfer-Learning-Model-in-Machine-Learning)

---

## Project Overview

This project demonstrates an image classification system using deep learning techniques to identify different jellyfish species. It utilizes transfer learning through the pre-trained VGG16 model from Keras Applications. The goal is to achieve high classification accuracy with a relatively small dataset by leveraging pre-trained weights.

---

## Dataset Description

- **Total Images:** 750  
- **Classes:** 5  
  - `barrel_jellyfish`  
  - `blue_jellyfish`  
  - `compass_jellyfish`  
  - `lions_mane_jellyfish`  
  - `mauve_stinger_jellyfish`  
- **Images per Class:** 150  
- **Image Size:** 224x224  
- **Batch Size:** 32  
- **Train/Test Split:**  
  - **Training:** 80% (19 batches)  
  - **Testing:** 20% (5 batches)

The dataset is stored in Google Drive and loaded using `image_dataset_from_directory()` from TensorFlow.

---

## Data Augmentation and Preprocessing

To reduce overfitting and improve generalization, the following augmentation techniques were used:

- Random horizontal flip  
- Random rotation (±0.2 radians)  
- Random zoom  
- Random brightness  
- Random contrast  

Data loading is optimized using TensorFlow’s `AUTOTUNE` to prefetch batches during training.

---

## Model Architecture

The model uses the VGG16 architecture from Keras, excluding its top classification layers:

python
model = Sequential([
    base_model,  # VGG16 base (frozen)
    Flatten(),
    Dense(256, activation='relu'),
    Dense(5, activation='softmax')  # 5 jellyfish classes
])

## Model Summary

- **Base Model:** VGG16 (pre-trained on ImageNet)  
- **Optimizer:** Adam  
- **Loss Function:** Sparse Categorical Crossentropy  
- **Evaluation Metric:** Accuracy  

The base model's weights are **frozen** to retain its learned image features, and only the custom classifier layers are trained.

---

## Training Results

The model was trained for **20 epochs**. Performance improved as follows:

- **Initial Training Accuracy (Epoch 1):** 37.05%  
- **Final Training Accuracy (Epoch 20):** 96.98%  
- **Test Accuracy:** 96.46%  
- **Test Loss:** 0.0781  

These results demonstrate the effectiveness of transfer learning in small dataset scenarios.

---

## Visualizations

### Sample Images  
A grid of 25 augmented training images was visualized with their associated class labels.

### Accuracy Plot  
A line chart showing steady improvement in training accuracy over 20 epochs was generated using Matplotlib.

---

## Conclusion

This project confirms that **transfer learning with VGG16** is an effective method for image classification with small datasets.  
With proper **data augmentation** and a **frozen convolutional base**, the model was able to achieve a test accuracy of over **96%**.  
This makes it suitable for deployment in **practical applications** involving image classification tasks.

