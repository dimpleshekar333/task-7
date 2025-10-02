# 🧬 Breast Cancer Classification using SVM

This repository contains a Python implementation of a **Support Vector
Machine (SVM)** classifier to predict whether a tumor is **benign** or
**malignant** using the Breast Cancer dataset.

------------------------------------------------------------------------

## 📌 Project Overview

This project demonstrates: - 📂 Loading the dataset from a CSV file. -
🧹 Preprocessing the data (encoding categorical labels, scaling
features). - 🤖 Training an **SVM classifier** (`linear` kernel by
default). - 📈 Evaluating the model (accuracy, confusion matrix,
classification report). - 🎨 Visualizing the confusion matrix using
**Seaborn heatmap**.

------------------------------------------------------------------------

## 📂 Dataset

You will need a dataset file named `breast-cancer.csv`.

👉 You can download a commonly used dataset here:\
[UCI Breast Cancer Wisconsin (Diagnostic)
Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

### Expected Columns

-   **Features** → numerical tumor measurements (radius, texture,
    perimeter, etc.)
-   **Target Column** → `"diagnosis"`
    -   `M` = Malignant (cancerous)\
    -   `B` = Benign (non-cancerous)

⚠️ If your dataset has a different target column name, update this line
in the script:

``` python
target_col = "diagnosis"
```

------------------------------------------------------------------------

## ⚙️ Installation

Install dependencies with:

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

------------------------------------------------------------------------

## 🚀 Usage

1.  Clone the repository:

    ``` bash
    git clone https://github.com/YourUsername/breast-cancer-svm.git
    cd breast-cancer-svm
    ```

2.  Place your dataset file in the correct path, e.g.:

        C:\Users\YourUsername\Downloads\breast-cancer.csv

3.  Update the file path in the script:

    ``` python
    file_path = r"C:\Users\YourUsername\Downloads\breast-cancer.csv"
    ```

4.  Run the script:

    ``` bash
    python task7.py
    ```

------------------------------------------------------------------------

## 📊 Example Output

Sample output:

    Dataset Shape: (569, 31)
    Accuracy: 0.97

Confusion Matrix Heatmap:

![Confusion Matrix
Example](https://scikit-learn.org/stable/_images/sphx_glr_plot_confusion_matrix_001.png)

------------------------------------------------------------------------

## 🛠️ Technologies Used

-   Python 3.x
-   [Pandas](https://pandas.pydata.org/)
-   [NumPy](https://numpy.org/)
-   [Matplotlib](https://matplotlib.org/)
-   [Seaborn](https://seaborn.pydata.org/)
-   [Scikit-learn](https://scikit-learn.org/)

------------------------------------------------------------------------

## 📌 Notes

-   You can experiment with different kernels (`linear`, `rbf`, `poly`)
    in SVM:

    ``` python
    svm_model = SVC(kernel='rbf', random_state=42)
    ```

-   Tune hyperparameters (`C`, `gamma`) for better results.

------------------------------------------------------------------------

## 📜 License

This project is licensed under the MIT License -- feel free to use and
modify it for learning purposes.
