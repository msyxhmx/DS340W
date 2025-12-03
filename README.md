# Quantifying the Impact of Genetic Algorithm-Based Feature Selection in Credit Card Fraud Detection

This repository contains the complete code and documentation for the research project focused on enhancing credit card fraud detection using supervised machine learning models in conjunction with Genetic Algorithm (GA) feature selection.

The primary artifact for this project is the Jupyter Notebook, which implements the entire experimental workflow, from data preprocessing and balancing to GA-based feature selection, model training, and comparative analysis.

## Getting Started

Follow these steps to set up the environment and run the code.

### 1\. Prerequisites

Ensure you have Python installed (version 3.8+ is recommended). The code runs best in a Jupyter environment (Notebook or Lab).

**_Note:_** We use Google Colab as our main environment

### 2\. Clone or Download the Repository

Clone this repository to your local machine

OR

Download as ZIP and extract manually.

### 3\. Open File

Click file named "Final Code_SyahmiMuslim_DanielShamsul.ipynb" and open it

### 4\. Install Dependencies

All necessary libraries are installed within the first code cell of the notebook. However, it's good practice to set up your environment beforehand.

Once activated, you can install the core packages:

pip install pandas numpy scikit-learn jupyter  

The notebook will handle the installation of specialized libraries like pygad, imbalanced-learn, catboost, and deap using !pip install commands.

### 5\. Data Setup

The notebook assumes the credit card transaction dataset is available.

- **Dataset:** This project utilizes the **Kaggle European Credit Card Fraud Detection Dataset**.
- **Action:** Ensure the dataset file (typically named train.csv or test.csv) is downloaded and placed directly in the **root directory** of this repository.

The first data-related code cell loads the data using the following path:

df = pd.read_csv('train.csv')

df = pd.read_csv('test.csv')

## Notebook Navigation and Execution

The entire project workflow is contained within a single file: **Final Code_SyahmiMuslim_DanielShamsul.ipynb**.

The notebook is structured into clear, sequential sections. **You must run the cells in order from top to bottom.**

## Detailed Project Workflow and Runtime

This section details the 11-step analytical pipeline implemented in the notebook. Running the cells sequentially ensures all steps are completed.

| **Step** | **Action/Notebook Title** | **Description** | **Estimated Runtime** |
| --- | --- | --- | --- |
| **1** | **Import Necessary Libraries** | Installs and imports all required Python packages (including pygad, imbalanced-learn, and catboost). | ~1 minute |
| --- | --- | --- | --- |
| **2** | **Load Data** | Loads train.csv and test.csv into a pandas DataFrame and performs initial cleaning. | Instant |
| --- | --- | --- | --- |
| **3** | **Feature Engineering** | _This step is implicitly handled during data loading and preparation, as the original V-features are already engineered._ | Instant |
| --- | --- | --- | --- |
| **4** | **Data Preprocessing (Downsampling and Scaling)** | Handles feature scaling (Amount and Time) and sets up the X (features) and y (target) variables. | Instant |
| --- | --- | --- | --- |
| **5** | **Data Balancing (SMOTE-Tomek)** | Splits the data and applies the **SMOTE-Tomek** technique to address the class imbalance, creating the final balanced training data. | **~2-5 minutes** |
| --- | --- | --- | --- |
| **6** | **ML Baseline Models Training** | Trains the initial models (Logistic Regression, Decision Tree, Random Forest) using **all original features** to establish the performance benchmark. | ~1 minute |
| --- | --- | --- | --- |
| **7** | **Machine Learning Models Training (Using GA-RF selected features)** | Retrains the Baseline Models (LR, DT, RF) using the **reduced feature subset** identified by the GA-RF. | ~1 minute |
| --- | --- | --- | --- |
| **8** | **GA Feature Selection (Novelty 1: F2-Score Fitness)** | Defines and executes the Genetic Algorithm using **Random Forest** as the fitness evaluator, optimized for the **F2-Score** (to increase Recall). | **~15-20 minutes** |
| --- | --- | --- | --- |
| **9** | **GA Feature Selection (Novelty 2: CatBoost Fitness)** | Defines and executes the Genetic Algorithm using the robust **CatBoost** model as the fitness evaluator. | **~15-20 minutes** |
| --- | --- | --- | --- |
| **10** | **Machine Learning Models Training (Using GA-CatBoost selected features)** | Retrains the Baseline Models (LR, DT, RF) and trains the **Final CatBoost Model** using the feature subset identified by the GA-CatBoost. | ~1-2 minutes |
| --- | --- | --- | --- |
| **11** | **Final Comparison of All ML Models** | Generates the final and sorted comparison table summarizing the performance of all 11 models across the baseline and GA-optimized phases. | Instant |
| --- | --- | --- | --- |

**_Note on Runtime:_** The Genetic Algorithm execution is computationally intensive and requires significant CPU time. The runtime listed is an estimate.

## Contribution

This project was developed by Syahmi Muslim and Daniel Shamsul.
