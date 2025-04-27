# Automated Classification of Lung and Colon Cancer from Histopathological Images

## About This Project

* **Problem Description-**
Histopathological analysis is critical for cancer diagnosis, where pathologists examine tissue samples under a microscope to detect and classify cancerous cells. This process faces challenges like time constraints, expert dependency, and inter-observer variability. In 2020, lung and colon cancers affected 4.19 million people worldwide, resulting in 2.7 million deaths. The shortage of pathologists, especially in developing regions, creates bottlenecks in timely diagnosis.
AI can transform this field by providing rapid, consistent preliminary assessments. Recent advancements in machine learning techniques have demonstrated promising results for histopathological image analysis. This project aims to develop an automated classification system that can accurately distinguish between benign and malignant tissues in lung and colon samples, potentially supporting early detection and reducing diagnostic delays that impact patient outcomes


* **Model Architecture-**
The model used is a custom Convolutional Neural Network (CNN) defined in `model.py`, designed specifically for this histopathology classification task. The architecture consists of the following key components:
    1.  **Input:** Accepts images with the number of channels specified in `config.INPUT_CHANNELS` .
    2.  **Convolutional Blocks (4 Blocks):** The core of the network comprises four sequential convolutional blocks. Each block follows a similar pattern:
        * Two `Conv2d` layers with 3x3 kernels and padding=1. These layers extract features from the input. The number of output filters increases with each block (32 -> 32 in Block 1, 32 -> 64 in Block 2, 64 -> 128 in Block 3, 128 -> 256 in Block 4).
        * `BatchNorm2d` layers after each convolution to stabilize learning and improve convergence.
        * `ReLU` activation functions after each BatchNorm layer to introduce non-linearity.
        * A `MaxPool2d` layer (2x2 kernel, stride 2) at the end of each block to reduce spatial dimensions and provide some translation invariance.
        * `Dropout` layers after each pooling layer, with increasing probability (p=0 for Blocks 1 & 2, p=0.15 for Block 3, p=0.2 for Block 4) to regularize the network and prevent overfitting in deeper layers.
    3.  **Flattening:** After the final convolutional block, the feature maps are flattened into a 1D vector. The size of this vector is calculated dynamically based on the input image size (`config.IMG_SIZE`) and the effect of the pooling layers.
    4.  **Fully Connected Layers:**
        * A `Linear` layer maps the flattened features to an intermediate size of 512 nodes, followed by a `ReLU` activation.
        * A `Dropout` layer with a higher probability (p=0.4) is applied before the final classification layer for further regularization.
        * A final `Linear` layer maps the 512 features to the number of output classes (`config.NUM_CLASSES`, which is 5). This layer outputs the raw logits for each class.

This architecture progressively extracts more complex features while reducing spatial dimensions and uses techniques like BatchNorm and Dropout to facilitate stable training and improve generalization on the histopathology images.

## Dataset Used

* **Dataset:** I have used the "Lung and Colon Cancer Histopathological Images" dataset.
* **Download:** It can be found on [Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) or the original [GitHub repo](https://github.com/tampapath/lung_colon_image_set).
* **Setup:**
    1.  Download the data.
    2. The Dataset folder inside is organised in such a way 
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
* **Citation:**
    > Borkowski AA, Bui MM, Thomas LB, Wilson CP, DeLand LA, Mastorides SM. Lung and Colon Cancer Histopathological Image Dataset (LC25000). arXiv:1912.12142 \[eess.IV], 2019.

## Installation

1.  **Clone:** Get the project files.
    ```bash
    git clone https://github.com/anuvindpramod/project_anuvind_pramod.git
    cd project_anuvind_pramod
    ```
2.  **Dataset:** Set up the `Dataset` folder as described above.
3.  **Environment:** Create and activate a Python environment (e.g., using `conda` or `venv`).
4.  **Install:** Use the provided `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## How the model was Trained:

I used a two-stage approach:

1.  **Initial Learning:** First, I tried to focus on getting the model to learn the basic patterns, so I tried an approach without any kind of strong regularization like augmentations, to check its capacity. (This will require manual setup if repeating).
2.  **Fine-tuning:** After the model was pre-trained in such a way, the model was then finetuned further carefully using the code in `train.py`. This involved loading the weights from the best performing model from the pre-training step and then using them along with data augmentation (like flips and rotations defined in `dataset.py`), using a lower learning rate, and employing early stopping to get the best generalization performance based on validation metrics. Checkpoints are saved in the `_checkpoints` directory.

## How to Run

1.  **Train:** Start the training process (uses settings from `config.py`).
    ```bash
    python train.py
    ```
2.  **Test:** Evaluate the best model on the test set.
    ```bash
    python test.py
    ```
3.  **Predict:** Run prediction on sample images placed in the `data/` folder). The data/ folder has all 5 class folders with each class folder containing 10 images of that class
    ```bash
    python predict.py
    ```


## Evaluation Metrics

The performance of the model is evaluated using the following metrics (calculated in `train.py`'s `calculate_and_print_metrics` function and used by `test.py`):

* **Overall Accuracy:** The proportion of correctly classified images out of the total.
* **Precision, Recall (Sensitivity), F1-Score:** Calculated per class, as macro averages, and as weighted averages. These metrics assess the model's ability to correctly identify positive instances and avoid false positives/negatives for each category.
* **Confusion Matrix:** A table showing the actual vs. predicted class counts, revealing specific misclassification patterns.
* **Specificity:** Calculated per class, measuring the model's ability to correctly identify true negative instances.
* **AUC (Area Under the ROC Curve):** Calculated using the One-vs-Rest (OvR) strategy with macro averaging, providing a measure of the model's ability to distinguish between classes across different thresholds.

## Test Set Results

The model was evaluated on a held-out test set comprising 5000 images (20% of the total dataset, 1000 images per class). The evaluation script (`test.py`) loaded the best weights saved during training (`_checkpoints/best_val_weights.pth`).

Key results on the test set are:

* **Overall Accuracy:** **0.9958** (99.58%)
* **Macro Average F1-Score:** **0.9958**
* **AUC (Macro OVR):** **0.9999**

**Performance Highlights:**

* The model achieved near-perfect classification for `lung_n`, `colon_aca`, and `colon_n` classes (1.000 F1-score).
* Performance for `lung_aca` (F1: 0.9894) and `lung_scc` (F1: 0.9896) was also very high.
* Specificity was excellent across all classes (ranging from 0.9958 to 1.0000).
* The confusion matrix showed minimal errors: the primary confusion was between `lung_aca` and `lung_scc` (17 `lung_aca` predicted as `lung_scc`, 4 `lung_scc` predicted as `lung_aca`).

These results indicate that the trained model generalizes very well to unseen data, achieving high accuracy and discriminative power across all five histopathological categories.

## Project Files Overview

* `config.py`: Key settings and hyperparameters.
* `dataset.py`: Handles loading and transforming the data.
* `model.py`: Defines the CNN architecture.
* `train.py`: Runs the model training loop.
* `test.py`: Evaluates the model on the test set.
* `predict.py`: Predicts classes for given image paths.
* `interface.py`: Standardizes names for grading compatibility.
* `_checkpoints/`: Stores saved model weights.
* `data/`: The data/ folder has all 5 class folders with each class folder containing 10 images of that class, this will be used for running the predict.py
* `Dataset/`: Where the full dataset is placed.
* `requirements.txt`: List of Python packages used.
* `README.md`
