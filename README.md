
# Face Expression Recognition using CNN

## Overview
This project implements a **Convolutional Neural Network (CNN)** for recognizing facial expressions. It uses the [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/ashishpatel26/face-expression-recognition-dataset) and detects seven basic emotions from grayscale images: *Anger*, *Disgust*, *Fear*, *Happiness*, *Sadness*, *Surprise*, and *Neutral*.

The CNN is built using **Keras** and **TensorFlow** and includes various layers like Convolution, MaxPooling, Dropout, and Fully Connected Layers. The model is trained on a dataset of 48x48 pixel grayscale images and evaluated on validation data to recognize facial emotions accurately.

## Features
- **Image Preprocessing**: Grayscale conversion, image resizing, and data augmentation.
- **Deep Learning Model**: A CNN with multiple convolutional and fully connected layers to extract features from images.
- **Training and Validation**: Real-time data generation and training with early stopping and learning rate reduction.
- **Visualization**: Training loss, validation loss, and accuracy plots are generated after model training.

## Technologies Used
- **Python 3.x**
- **TensorFlow / Keras**: For building and training the CNN.
- **OpenCV**: For image manipulation.
- **Matplotlib / Seaborn**: For visualizing data and results.
- **NumPy / Pandas**: For data handling and processing.
- **PIL (Pillow)**: For image loading and manipulation.

## Dataset
The dataset used is the [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/ashishpatel26/face-expression-recognition-dataset), which consists of images categorized into seven emotion classes: *Angry*, *Disgust*, *Fear*, *Happy*, *Neutral*, *Sad*, and *Surprise*.

The dataset is divided into **train** and **validation** directories, and images are resized to 48x48 pixels.

## Installation and Setup
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/face-expression-recognition.git
cd face-expression-recognition
```

### 2. Install Dependencies
Ensure that you have Python 3.x installed, then install the required libraries:
```bash
pip install -r requirements.txt
```

### 3. Dataset Preparation
- Download the dataset from [Kaggle](https://www.kaggle.com/datasets/ashishpatel26/face-expression-recognition-dataset).
- Place the dataset in the project folder with the following structure:
  ```
  /kaggle/input/face-expression-recognition-dataset/images
  ├── train
  └── validation
  ```

### 4. Running the Model
To train the model, run:
```bash
python train_model.py
```

This script loads the dataset, builds the CNN model, and starts training.

### 5. Model Output
- The model will be saved as `Emotion_recognition.h5` after training.
- The training and validation loss and accuracy will be plotted after the training process.

## Model Architecture
The CNN architecture includes:
1. **Convolutional Layers**: Extract features from images using filters.
2. **MaxPooling Layers**: Reduce dimensionality by down-sampling.
3. **Batch Normalization**: Speed up training and stabilize the learning process.
4. **Dropout**: Prevent overfitting by randomly disabling neurons.
5. **Fully Connected Layers**: Learn complex representations for classifying emotions.

### Optimizer
- **Adam**: Optimizes the model with a learning rate of `0.0001`, using `categorical_crossentropy` as the loss function.

### Callbacks
- **EarlyStopping**: Stops training if the validation loss doesn't improve for 3 epochs.
- **ReduceLROnPlateau**: Reduces the learning rate when validation loss plateaus.
- **ModelCheckpoint**: Saves the best model based on validation accuracy.

## Results and Visualization
After training, the model's performance is visualized through loss and accuracy graphs:

- **Loss Curve**: Displays the training and validation loss over epochs.
- **Accuracy Curve**: Shows how the training and validation accuracy change over time.


![Training and Validation Loss/Accuracy](path_to_screenshot)

## Future Improvements
- Integrate more data augmentation techniques to improve model generalization.
- Experiment with different optimizers and learning rate schedules.
- Use transfer learning with a pre-trained model for better performance.

## License
This project is licensed under the MIT License.


