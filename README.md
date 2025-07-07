# 🍅 Tomato Leaf Disease Classification

## 🧠 Overview

This project implements a deep learning model using **InceptionV3** with transfer learning to classify various tomato leaf diseases. The model is trained on a dataset of tomato leaf images to recognize multiple disease types and healthy leaves.

---

## 📦 Requirements

To run this project, install the following dependencies:

- Python 3.6+
- TensorFlow 2.17.1
- NumPy
- Matplotlib
- Glob

Install with:

```bash
pip install tensorflow numpy matplotlib

⚠️ Note: tensorflow-gpu is not used due to installation issues in the notebook. If you have a GPU, ensure CUDA and cuDNN are properly configured or proceed with the CPU version.

🗂️ Dataset Structure
The dataset should be organized as follows:


datasets/
├── train/
│   ├── Tomato___Late_blight/
│   ├── Tomato___Early_blight/
│   ├── Tomato___Spider_mites Two-spotted_spider_mite/
│   ├── Tomato___Target_Spot/
│   ├── Tomato___Tomato_mosaic_virus/
│   ├── Tomato___Leaf_Mold/
│   ├── Tomato___Bacterial_spot/
│   ├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
│   ├── Tomato___Septoria_leaf_spot/
│   └── Tomato___healthy/
└── valid/
    ├── Tomato___Late_blight/
    ├── Tomato___Early_blight/
    └── ...
Training set: 18,345 images across 10 classes

Validation set: 4,588 images across 10 classes

📍 Make sure the dataset is available at:

'''bash

/content/drive/MyDrive/datasets/train  
/content/drive/MyDrive/datasets/valid
Or update the paths in the code as needed.

🏗️ Model Architecture
Base Model: InceptionV3 pre-trained on ImageNet, with top layers removed.

Custom Layers:

Flatten layer

Dense layer with 10 units (classes) and softmax activation

➡️ All InceptionV3 layers are frozen (layer.trainable = False) to leverage transfer learning.

🏋️ Training
Image Preprocessing:

Resize to 224x224

Augmentation: shear, zoom, horizontal flip

Rescale pixel values to [0, 1]

Configuration:

Optimizer: Adam

Loss: Categorical Crossentropy

Metrics: Accuracy

Epochs: 10

Batch Size: 16 (train), 32 (valid)

Results:

Final Training Accuracy: ~85.47%

Final Validation Accuracy: ~83.26%

Loss/Accuracy plots saved as:

LossVal_loss.png

AccVal_acc.png

🚀 Usage
⚙️ Environment Setup
Install dependencies

Download the dataset

Verify paths in the notebook

📓 Running the Notebook
Open tomato_leaf_disease_classification.ipynb

Run each cell to:

Install packages (if needed)

Load & preprocess data

Build, compile, and train model

Save model (model_inception.h5)

Visualize training metrics

🔍 Making Predictions
To make predictions with a new image:

Load the saved model:


from tensorflow.keras.models import load_model
model = load_model('model_inception.h5')
Preprocess your image:

Resize to 224x224

Normalize and apply InceptionV3 preprocessing

Predict:


import numpy as np
prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
✅ Example code is available in the notebook.

📝 Notes
Notebook was run on Google Colab

GPU was not detected; adjust for local environments if needed

tensorflow-gpu installation failed; used tensorflow==2.17.1 instead

Model saved in HDF5 format. Consider migrating to .keras in future versions.

📄 License
This project is licensed under the MIT License.



---

### ✅ Next Steps

- Save this content as `README.md` in your project root.
- Optionally include badges (Colab, License, etc.) or screenshots of plots in future improvements.

Let me know if you'd like me to generate a LICENSE file or badge section as well.
