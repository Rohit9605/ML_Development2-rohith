
# EEG Classification with Residual CNN

This project implements a Residual Convolutional Neural Network (CNN) to classify EEG signals as either left or right hand movements. It includes data preprocessing, model architecture definition, training, and evaluation scripts split across modular files.

---

## Project Overview

The codebase is organized into four main components:

- **preprocess.py**  
  Handles extraction of pickle-based EEG data, applies signal filtering and segmentation, reorders channels, and prepares train/validation/test splits as PyTorch datasets.

- **model.py**  
  Defines a Residual CNN model using PyTorch’s `nn.Module`. The architecture includes residual blocks with Conv1D layers and ReLU activations.

- **train.py**  
  Trains the model using the preprocessed data. Includes a training loop with backpropagation, optimizer setup, and epoch-wise progress display.

- **evaluate.py**  
  (To be implemented) Will include evaluation logic for assessing model performance on validation and test sets.

---

## Preprocessing Steps

1. **Data Extraction:**  
   EEG recordings are stored in `.pkl` files inside `LHNT_EEG.zip`. The code unzips the file and finds all relevant pickle files organized in folders.

2. **Labeling:**  
   Each pickle file is loaded and assigned a label based on the file name—`1` for "right" and `0` for all other movements.

3. **Bandpass Filtering:**  
   A 4th-order Butterworth bandpass filter is applied in the frequency range of 1–40 Hz to remove noise and retain key EEG frequency bands.

4. **Channel Reordering:**  
   Channels are reordered into a consistent format using a predefined list of indices.

5. **Segmentation:**  
   Signals are divided into overlapping 1-second windows with a 16ms shift to generate more data samples from continuous recordings.

6. **Dataset Splitting:**  
   The processed signals and corresponding labels are split into training, validation, and testing sets. These are then converted into PyTorch `TensorDataset` objects and wrapped in DataLoaders for training.

---

## Model Architecture

The model is a Residual CNN designed to process 1D EEG signal segments.

- **Residual Block:**  
  Each residual block includes two Conv1D layers followed by ReLU activations. A skip connection allows the input to bypass the convolutions and be added to the output.

- **Network Structure:**  
  - Input layer: Conv1D on the EEG channels.
  - Two residual blocks: Improve learning and allow deeper architectures.
  - Flatten layer: Converts output to 2D.
  - Fully connected layer: Outputs a 2-class prediction (left/right).

---

## Training Procedure

- **Optimizer:** Adam  
- **Loss Function:** Cross-Entropy Loss  
- **Epochs:** Defined in code (e.g., 30 or 50 based on hyperparameters)  
- **Training Loop:**
  - Loads batches from training DataLoader.
  - Performs forward pass, computes loss, and applies backpropagation.
  - Prints running loss and epoch statistics.

---

## Evaluation Criteria

The evaluation step (to be implemented) will:
- Load trained model weights.
- Perform predictions on validation and test sets.
- Compute accuracy and potentially confusion matrix or F1-score.

---

## Dependencies

To run the project:

```bash
pip install torch numpy scipy scikit-learn matplotlib tqdm
```

---

## How to Run

```bash
# Preprocessing
python preprocess.py

# Training
python train.py

# Evaluation (future)
python evaluate.py
```

