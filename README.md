# Breast Cancer Classification using Neural Networks

## Overview
This project implements a neural network model for breast cancer classification using a dataset containing various features related to tumor characteristics. The model is built using Python and deep learning frameworks such as TensorFlow/Keras. The goal is to develop a reliable and accurate classification system to assist in early diagnosis.

## Dataset
The dataset used for this project consists of labeled breast cancer tumor data, sourced from publicly available repositories such as the UCI Machine Learning Repository. It includes features extracted from digitized images of fine needle aspirate (FNA) of breast masses, describing characteristics of the cell nuclei present in the images.

### Features of the Dataset
- **Mean Radius**: The mean size of the tumor cell nuclei.
- **Mean Texture**: The variation in texture of the cell nuclei.
- **Mean Perimeter**: The mean perimeter of the cell nuclei.
- **Mean Area**: The mean area of the cell nuclei.
- **Mean Smoothness**: The degree of smoothness of the cell nuclei.
- **Other features**: Compactness, Concavity, Symmetry, and Fractal Dimension.

The dataset is preprocessed before training to ensure optimal performance.

## Features
- Neural network-based classification using TensorFlow/Keras.
- Data preprocessing including normalization and feature selection.
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix.
- Visualization of data insights and model performance.
- Hyperparameter tuning for model optimization.

## Installation
To set up the project environment, install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Running the Jupyter Notebook
Run the Jupyter Notebook to train and evaluate the model:
```bash
jupyter notebook Breast_Cancer_classification_NN_Model.ipynb
```

### Steps to Execute
1. Load and explore the dataset.
2. Preprocess the data (handling missing values, normalization, and feature selection).
3. Split the dataset into training and testing sets.
4. Build and compile the neural network model.
5. Train the model and evaluate its performance.
6. Visualize results and analyze model performance.

## Model Architecture
The neural network consists of multiple layers including:
- **Input Layer**: Accepts feature inputs.
- **Hidden Layers**: Fully connected layers with activation functions such as ReLU.
- **Output Layer**: Uses a sigmoid activation function for binary classification.

## Evaluation Metrics
The model is evaluated using the following metrics:
- **Accuracy**: Measures overall correctness.
- **Precision**: Measures the proportion of correctly identified positive cases.
- **Recall**: Measures the proportion of actual positive cases correctly identified.
- **F1-score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: Visual representation of model predictions vs actual labels.

## Results
The trained model achieves a high classification accuracy, demonstrating its effectiveness in distinguishing between malignant and benign tumors. The confusion matrix and ROC curve provide insights into model performance.

## Hyperparameter Tuning
Various hyperparameters such as learning rate, batch size, number of hidden layers, and activation functions are optimized to enhance model performance.

## Contributing
Contributions are welcome! If you would like to improve this project, feel free to fork the repository, make enhancements, and submit a pull request.

## License
This project is licensed under the MIT License.

## Acknowledgments
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)

