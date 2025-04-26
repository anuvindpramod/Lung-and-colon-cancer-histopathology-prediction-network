# Automated Classification of Lung and Colon Cancer from Histopathological Images

## About This Project

* **Problem Description-**
Histopathological analysis is critical for cancer diagnosis, where pathologists examine tissue samples under a microscope to detect and classify cancerous cells. This process faces challenges like time constraints, expert dependency, and inter-observer variability. In 2020, lung and colon cancers affected 4.19 million people worldwide, resulting in 2.7 million deaths. The shortage of pathologists, especially in developing regions, creates bottlenecks in timely diagnosis.
AI can transform this field by providing rapid, consistent preliminary assessments. Recent advancements in machine learning techniques have demonstrated promising results for histopathological image analysis. This project aims to develop an automated classification system that can accurately distinguish between benign and malignant tissues in lung and colon samples, potentially supporting early detection and reducing diagnostic delays that impact patient outcomes

* **Model Architecture and Reasoning-**
The proposed custom convolutional neural network architecture will employ sequential convolutional blocks using 3×3 filters to capture delicate histopathological patterns, integrated with batch normalization for training stability and ReLU activation functions for non-linear feature learning, followed by max-pooling layers for progressive dimension reduction. The architecture could incorporate dropout layers (20-30% rates) between convolutional blocks and implement L2 regularization on kernel weights to mitigate overfitting. The classification head utilizes global average pooling to condense spatial features while preserving critical diagnostic information, followed by densely connected layers with progressively reduced units to distill hierarchical representations, culminating in a 5-unit softmax output layer for final class probability distribution. This design would enable specialized optimization for detecting subtle malignant tissue abnormalities in histopathological images while maintaining computational efficiency through deliberate layer depth and parameter constraints and avoiding biases inherent in pre-trained models developed for non-medical image domains.

## Dataset Used: LC25000

* **Dataset:** We used the "Lung and Colon Cancer Histopathological Images" (LC25000) dataset.
* **Download:** You can find it on [Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) or the original [GitHub repo](https://github.com/tampapath/lung_colon_image_set).
* **Setup:**
    1.  Download the data.
    2.  Create a `Dataset` folder in your project root.
    3.  Organize the images inside `Dataset` like this:
        ```
        Dataset/
        ├── colon_image_sets/
        │   ├── colon_aca/*.jpeg
        │   └── colon_n/*.jpeg
        └── lung_image_sets/
            ├── lung_aca/*.jpeg
            ├── lung_n/*.jpeg
            └── lung_scc/*.jpeg
        ```
* **Citation:** Please cite the original authors if you use this dataset:
    > Borkowski AA, Bui MM, Thomas LB, Wilson CP, DeLand LA, Mastorides SM. Lung and Colon Cancer Histopathological Image Dataset (LC25000). arXiv:1912.12142 \[eess.IV], 2019.

## Installation

1.  **Clone:** Get the project files.
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
2.  **Dataset:** Set up the `Dataset` folder as described above.
3.  **Environment:** Create and activate a Python environment (e.g., using `conda` or `venv`).
4.  **Install:** Use the provided `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## How We Trained The Model

We used a two-stage approach:

1.  **Initial Learning:** First, we focused on getting the model to learn the basic patterns, possibly without strong regularization, to check its capacity. (This might require manual setup if repeating).
2.  **Fine-tuning:** We then trained the model more carefully using the code in `train.py`. This involved loading the initial weights (if available), adding data augmentation (like flips and rotations defined in `dataset.py`), using a lower learning rate, and employing early stopping to get the best generalization performance based on validation metrics. Checkpoints are saved in the `_checkpoints` directory.

## How to Run

1.  **Train:** Start the training process (uses settings from `config.py`).
    ```bash
    python train.py
    ```
2.  **Test:** Evaluate the best model on the test set.
    ```bash
    python test.py
    ```
3.  **Predict (Example):** Run prediction on sample images placed in the `data/` folder (as required for submission [cite: 16, 17]).
    ```bash
    python predict.py
    ```
    *(Remember to put 10 sample images per class in the `data/` directory before running this for submission examples)*

## Project Files Overview

* `config.py`: Key settings and hyperparameters.
* `dataset.py`: Handles loading and transforming the data.
* `model.py`: Defines the CNN architecture.
* `train.py`: Runs the model training loop.
* `test.py`: Evaluates the model on the test set.
* `predict.py`: Predicts classes for given image paths.
* `interface.py`: Standardizes names for grading compatibility.
* `_checkpoints/`: Stores saved model weights.
* `data/`: Sample images required for submission[cite: 6].
* `Dataset/`: Where the full LC25000 dataset should be placed.
* `requirements.txt`: List of Python packages needed.
* `README.md`: This file.
