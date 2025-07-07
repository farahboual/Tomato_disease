Tomato Leaf Disease Classification
Overview
This project implements a deep learning model for classifying tomato leaf diseases using the InceptionV3 architecture with transfer learning. The model is trained on a dataset of tomato leaf images to identify various diseases and healthy leaves.
Requirements
To run this project, you need the following dependencies:

Python 3.6+
TensorFlow 2.17.1
NumPy
Matplotlib
Glob

You can install the required packages using:
pip install tensorflow numpy matplotlib

Note: The tensorflow-gpu package is not used due to installation issues observed in the notebook. Ensure you have a compatible version of TensorFlow installed. If you have a GPU, ensure CUDA and cuDNN are properly configured, or use the CPU version as shown in the notebook.
Dataset
The dataset is expected to be organized in the following structure:
/datasets/
    train/
        Tomato___Late_blight/
        Tomato___Early_blight/
        Tomato___Spider_mites Two-spotted_spider_mite/
        Tomato___Target_Spot/
        Tomato___Tomato_mosaic_virus/
        Tomato___Leaf_Mold/
        Tomato___Bacterial_spot/
        Tomato___Tomato_Yellow_Leaf_Curl_Virus/
        Tomato___Septoria_leaf_spot/
        Tomato___healthy/
    valid/
        Tomato___Late_blight/
        Tomato___Early_blight/
        ...


Training set: Contains 18,345 images across 10 classes.
Validation set: Contains 4,588 images across 10 classes.
Ensure the dataset is accessible at the specified paths (/content/drive/MyDrive/datasets/train and /content/drive/MyDrive/datasets/valid) or update the paths in the code accordingly.

Model Architecture

Base Model: InceptionV3 pre-trained on ImageNet, with the top layer removed.
Custom Layers: 
Flatten layer to convert the output of InceptionV3 to a 1D vector.
Dense layer with 10 units (corresponding to 10 classes) and softmax activation.


The pre-trained weights of InceptionV3 are frozen (layer.trainable = False) to leverage transfer learning.

Training

Image Preprocessing:
Images are resized to 224x224 pixels.
Data augmentation (shear, zoom, horizontal flip) is applied to the training set to improve generalization.
Pixel values are rescaled to [0, 1] using rescale=1./255.


Training Configuration:
Optimizer: Adam
Loss: Categorical Crossentropy
Metrics: Accuracy
Epochs: 10
Batch Size: 16 (training), 32 (validation)


Training Results:
Final training accuracy: ~85.47%
Final validation accuracy: ~83.26%
Loss and accuracy plots are generated and saved as LossVal_loss.png and AccVal_acc.png.



Usage

Setup the Environment:

Install the required dependencies.
Ensure the dataset is available and paths are correctly set in the notebook.


Running the Notebook:

Open the Jupyter notebook (tomato_leaf_disease_classification.ipynb).
Execute the cells sequentially to:
Install dependencies (if needed).
Load and preprocess the dataset.
Build and compile the model.
Train the model.
Save the trained model as model_inception.h5.
Visualize training results.




Making Predictions:

Load the saved model (model_inception.h5).
Preprocess a new image (resize to 224x224, normalize, and apply InceptionV3 preprocessing).
Use the model to predict the class of the image.
Example prediction code is included in the notebook.



Files

tomato_leaf_disease_classification.ipynb: The main Jupyter notebook containing the code.
model_inception.h5: The trained model file (generated after training).
LossVal_loss.png: Plot of training and validation loss.
AccVal_acc.png: Plot of training and validation accuracy.

Notes

The notebook was run on a Google Colab environment, as indicated by the paths (/content/drive/MyDrive/...). For local execution, update the dataset paths to match your file system.
The nvidia-smi command failed in the notebook, indicating no GPU was detected. Ensure GPU support is properly configured if you intend to use a GPU.
The tensorflow-gpu installation failed due to a metadata generation error. The notebook uses the standard tensorflow package (version 2.17.1), which worked successfully.
The model is saved in HDF5 format. Consider using the native Keras format (.keras) for future compatibility, as suggested by the warning in the notebook.

