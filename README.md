# Kernelized Linear Classification

This project demonstrates the implementation and evaluation of several machine learning algorithms for label classification including Perceptron, Pegasos for SVM, and regularized logistic classification based on numerical features. 
Since the underlying data is not linearly-separable, these baseline models are then enhanced by using feature-expansion and kernel methods. All the functions are implemented from scratch using `Python` in a jupter notebook environment.

> [!IMPORTANT]
> Preview the main project files below:
> 
> | File | Description | Link |
> |------|--------------|------|
> | `Kernelized_Linear_Classification.ipynb` | Python Codes for the Project | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tabaraei/Kernelized-Linear-Classification/blob/main/Kernelized_Linear_Classification.ipynb) |
> | `report.pdf` | Report Including the Experiment Setup | [![Paper](https://img.shields.io/badge/Report-PDF-red)](https://github.com/tabaraei/Kernelized-Linear-Classification/blob/main/report.pdf) |

## Installation

The project was developed using `Python 3.x` and within Google Colaboratory environment. In order to run the project in a jupyter notebook on local machine with `Windows` OS (with Python version `3.9.13`), please run the following commands respectively:
- Clone the project using `git clone https://github.com/tabaraei/Kernelized-Linear-Classification.git`, or directly download the zip file, and store the `Kernelized-Linear-Classification` codebase on desired directory in your local machine.
- Once the installation is performed, navigate to the codebase as `cd Kernelized-Linear-Classification`.
- Run the commands below sequentially to install the requirements, keeping in mind that the commands below will attempt to create a virtual Python environment in the project directory (in a subfolder named `venv`), and will install all the packages inside `requirements.txt` file accordingly:
  ```shell
  py -3 -m venv venv
  venv/Scripts/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
- Finally, open the codebase in the browser by running `jupyter notebook`, where you can view, edit, and run the `Kernelized-Linear-Classification.ipynb`.



## Project Structure

### 1. Data Loading and Exploration
- **Loading the Dataset and Libraries:** The project begins by loading the necessary libraries and the dataset, which includes 10,000 samples with 10 numerical features.
- **Data Distribution Exploration:** An exploration of the dataset to understand the distribution of features and target labels, which aids in deciding preprocessing steps.

### 2. Data Preprocessing
- **Presence of Missing Values:** The dataset is checked for missing values to ensure data integrity before analysis.
- **Check for Duplicates:** A check for duplicate entries in the dataset is conducted to avoid redundant data during model training.
- **Feature Selection:** Correlation analysis is performed to remove redundant features that do not contribute to the model's performance.
- **Feature Normalization:** Z-score normalization is applied to standardize the range of independent variables or features.
- **Partitioning the Dataset:** The dataset is split into training and testing sets to evaluate the performance of the models.

### 3. Hyper-parameter Tuning and Model Evaluation
- **K-fold Cross-Validation:** K-fold cross-validation is used to assess the performance of the models by dividing the dataset into K parts and training the model K times.
- **K-fold Nested Cross-Validation:** Nested cross-validation is implemented for hyper-parameter tuning to avoid overfitting and to provide an unbiased estimate of model performance.

### 4. Model Implementation
- **Perceptron:** The Perceptron algorithm is implemented as a baseline linear classifier to evaluate its performance on the dataset.
- **Pegasos for SVM:** The Pegasos algorithm is applied to implement Support Vector Machines, optimizing the model's performance on linear data.
- **Regularized Logistic Classification:** Logistic regression with regularization is used to improve classification accuracy by preventing overfitting.

### 5. Feature Expansion
- **Feature-Expanded Perceptron:** The Perceptron model is expanded with additional features to improve its ability to capture complex patterns in the data.
- **Feature-Expanded Pegasos for SVM:** The Pegasos SVM is enhanced by expanding features, which helps in better classification of non-linearly separable data.
- **Feature-Expanded Logistic Classification:** The logistic regression model is used with additional features to enhance its classification capabilities.

### 6. Kernel Methods
- **Kernelized Perceptron:** The Perceptron model is kernelized using polynomial and Gaussian kernels to handle non-linear separable data effectively.
- **Kernelized Pegasos for SVM:** The Pegasos algorithm is combined with kernel methods to enhance the SVM's performance on non-linear data.

## Image Gallery

| Feature's Distribution | Target's Distribution |
| --- | --- |
| ![Distribution of Features](https://github.com/tabaraei/Kernelized-Linear-Classification/blob/main/latex/images/distribution_features.png) | ![Distribution of Target Labels](https://github.com/tabaraei/Kernelized-Linear-Classification/blob/main/latex/images/distribution_targets.png) |

| Outlier Removal | Correlation Detection |
| --- | --- |
| ![Outliers After Removal](https://github.com/tabaraei/Kernelized-Linear-Classification/blob/main/latex/images/outliers_after.png) | ![Correlation Matrix](https://github.com/tabaraei/Kernelized-Linear-Classification/blob/main/latex/images/correlation.png) |

| Performance Measures | Runtimes |
| --- | --- |
| ![Model Performance](https://github.com/tabaraei/Kernelized-Linear-Classification/blob/main/latex/images/performance.png) | ![Runtime Analysis](https://github.com/tabaraei/Kernelized-Linear-Classification/blob/main/latex/images/runtime.png) |

## Conclusion

The project successfully demonstrated the effectiveness of using both feature-expanded and kernelized methods in enhancing the performance of baseline linear classifiers on the given non-linearly separable data, with the `Polynomial Kernel Perceptron` emerging as the top classifier with 94% average accuracy on test set.
