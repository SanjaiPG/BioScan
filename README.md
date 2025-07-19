# BioScan
This repository contains the implementation of a deep learning model built using TensorFlow and Keras for classifying plant diseases. The model utilizes a Convolutional Neural Network (CNN) architecture to recognize various plant diseases based on images from the "New Plant Diseases Dataset" sourced from Kaggle.

This model will be deployed within a local application, enabling users to classify plant diseases directly on their own devices without needing an internet connection.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Model Evaluation](#model-evaluation)
- [Accuracy Visualization](#accuracy-visualization)
- [Confusion Matrix](#confusion-matrix)
- [Saving the Model](#saving-the-model)
- [Local App Deployment](#local-app-deployment)

## Overview

This project aims to develop an efficient deep learning model for classifying images of plant diseases using Convolutional Neural Networks (CNNs). The dataset contains images of various plant diseases, which are preprocessed and fed into a CNN model for training. Once trained, this model can be deployed within a local application to allow users to classify plant diseases directly on their devices.

## Getting Started
Follow the steps below to run this project on your local machine.

# Clone this repository:


```bash
git clone https://github.com/SanjaiPG/BioScan.git
```

## Dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Download the dataset from Kaggle:

The dataset can be downloaded from Kaggle using the following command:

DataSet link
```bash
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
```

```bash
kaggle datasets download -d vipoooool/new-plant-diseases-dataset
```

Make sure you have your Kaggle credentials set up (username and key) in your environment variables.

Unzip the dataset:

bash
Copy
Edit
unzip new-plant-diseases-dataset.zip
Dependencies
tensorflow: Deep learning library used for building the CNN model.

matplotlib: For visualizing accuracy plots and confusion matrix.

pandas: Data manipulation.

seaborn: For enhanced data visualization, particularly confusion matrices.

scikit-learn: For generating the classification report.

To install the required dependencies, run:

bash
Copy
Edit
pip install tensorflow matplotlib pandas seaborn scikit-learn
Dataset
The dataset used in this project is the "New Plant Diseases Dataset (Augmented)" which contains images of plant leaves, each labeled with the type of disease or class. The images have been augmented to provide more diverse training data for better model generalization.

Training Data: /train

Validation Data: /valid

You can find more details on the dataset on Kaggle.

Model Architecture
The CNN model has the following architecture:

Convolutional Layers: The model consists of several convolutional layers (Conv2D) with ReLU activation, followed by max-pooling layers to reduce the spatial dimensions.

Dropout Layers: Dropout is applied after the dense layers to reduce overfitting.

Fully Connected Layers: The output is passed through dense layers before arriving at the final classification layer.

Model Layers
python
Copy
Edit
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[128, 128, 3]))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=1500, activation='relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(units=38, activation='softmax'))
Training the Model
Once the data is loaded, the model is trained using the following parameters:

Optimizer: Adam optimizer with a learning rate of 0.0001

Loss Function: Categorical Crossentropy

Metrics: Accuracy

python
Copy
Edit
trainingHistory = model.fit(x=training, validation_data=validation, epochs=10)
Model Evaluation
The trained model is evaluated on both the training and validation datasets, and the loss and accuracy values are printed:

python
Copy
Edit
loss, accuracy = model.evaluate(training)
valLoss, valAccuracy = model.evaluate(validation)
Accuracy Visualization
The accuracy of the model is plotted over the course of the training epochs. The plot compares training accuracy with validation accuracy.

python
Copy
Edit
plt.plot(epochs, trainingHistory.history['accuracy'], label='Training Accuracy')
plt.plot(epochs, trainingHistory.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
Confusion Matrix
After training, the model's performance is evaluated using a confusion matrix. This helps to visualize the performance in terms of true positives, false positives, true negatives, and false negatives.

python
Copy
Edit
from sklearn.metrics import confusion_matrix, classification_report
cn = confusion_matrix(y_true, predicted_cat)
sns.heatmap(cn, annot=True, annot_kws={'size':10})
Saving the Model
The trained model is saved in the .keras format for later use.

python
Copy
Edit
model.save("model.keras")
Local App Deployment
This model can be deployed within a local application, allowing users to run the model directly on their machines. The app can take images of plant leaves and classify the disease using the model, making it useful for farmers, researchers, and plant enthusiasts.

You can use tools like Flask, Streamlit, or PyQt to create the local app interface. The model file (model.keras) can be loaded locally, and the app will provide real-time predictions for plant diseases based on the uploaded image
