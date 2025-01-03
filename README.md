# Galaxy-ZOO-Image-Classifier-with-Deep-Learning

# Galaxy Image Classifier

## Overview
This project uses a **Convolutional Neural Network (CNN)** to classify galaxy images into five distinct categories based on the **Sloan Digital Sky Survey (SDSS)** data. The images are preprocessed and trained using a CNN model to achieve a high classification accuracy of **89.4%**.

## Data Source
The galaxy classification dataset was obtained from the following sources:
- **Kaggle Dataset:** [Galaxy Zoo Classification](https://www.kaggle.com/datasets/anjosut/galaxy-zoo-classification)
- **Download Link:** [Google Drive](https://drive.google.com/file/d/1_BhlRkWn-Yg-mcsOkPklhYoHbKyOX1qF/view?usp=sharing)

### Galaxy Classes
The dataset consists of five galaxy types:
- **Cigar-shaped smooth:** Class 0
- **In between smooth:** Class 1
- **Completely round smooth:** Class 2
- **Edge-on:** Class 3
- **Spiral:** Class 4

## Methodology

### Techniques Used
1. **Data Preprocessing:**
   - Loaded image paths and labels from a CSV file.
   - Verified image paths and filtered invalid entries.
   - Normalized image pixel values to the range [0, 1] using `ImageDataGenerator`.
   - Resized images to 128x128 pixels.
   - Split the dataset into training (80%), validation (5%), and testing (15%) subsets.

2. **Model Architecture:**
   - Built a sequential CNN with three convolutional layers, each followed by max-pooling.
   - Flattened the output of the convolutional layers and added fully connected dense layers.
   - Included a dropout layer (rate = 0.5) to prevent overfitting.
   - Used a softmax activation function in the output layer for multi-class classification.

3. **Training and Evaluation:**
   - Used `adam` optimizer with categorical cross-entropy as the loss function.
   - Trained the model for 10 epochs with batch size 32.
   - Visualized accuracy and loss trends for training and validation datasets.

4. **Performance Metrics:**
   - Calculated a test accuracy of **89.4%**.
   - Analyzed the model's performance using a confusion matrix and classification report.

### Model Summary
- **Input Shape:** (128, 128, 3)
- **Convolution Layers:**
  - Conv2D with 32, 64, and 128 filters, each with kernel size (3x3) and ReLU activation.
  - MaxPooling2D after each convolutional layer.
- **Fully Connected Layers:**
  - Dense layer with 128 units and ReLU activation.
  - Dropout layer with a rate of 0.5.
  - Output layer with 5 units (softmax activation).

### Visualization
- Confusion matrix and classification report highlight model performance.
- Accuracy and loss trends show consistent improvements over training epochs.

## Results
The CNN achieved an impressive accuracy of **89.4%**, demonstrating its effectiveness in classifying galaxy images into their respective categories.

---
For any questions or further information, feel free to reach out!

