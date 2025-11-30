# Custom Transformer Implementation

A fully functional Transformer model implemented from scratch using **PyTorch**. This project demonstrates the inner workings of the Transformer architecture, including self-attention, backpropagation, and optimization.

## Features

*   **PyTorch Implementation**: Utilizes PyTorch for tensor operations and automatic differentiation.
*   **Byte-Level Tokenization**: Operates directly on UTF-8 bytes (vocab size 260), allowing it to process any text string without a pre-computed vocabulary.
*   **Custom Components**: Implementation of core Transformer components (`MultiHeadAttention`, `Transformer`, `Encoder`, `Decoder`).
*   **Training Loop**: Complete training pipeline with `CrossEntropyLoss`.

## Project Structure

*   `config.py`: Configuration dataclass for model hyperparameters.
*   `layers.py`: Core neural network layers.
*   `attention.py`: Multi-Head Attention mechanism implementation.
*   `transformer.py`: Full Transformer architecture (`Encoder`, `Decoder`, `Transformer`).
*   `loss.py`: `CrossEntropyLoss` implementation.
*   `utils.py`: Helper functions for tokenization and masking.
*   `clean_dataset.py`: Script to clean and prepare the bilingual dataset.
*   `train.py`: Main training script.
*   `main.py`: Demo script for inference and shape verification.

## Requirements

*   Python 3.x
*   PyTorch
*   NumPy

```bash
pip install torch numpy
```

## Usage

### 1. Data Preparation

1.  Download the **Bilingual Sentence Pairs** dataset from Kaggle.
2.  Extract the downloaded archive into the root directory of this project.
3.  Ensure the folder is named `bilingual-sentence-pairs`.

The structure should look like this:
```
CustomTransformer/
├── bilingual-sentence-pairs/
│   ├── afr.txt
│   ├── ...
├── clean_dataset.py
├── train.py
...
```

4.  Run the cleaning script to format the data:

```bash
python clean_dataset.py
```
This will process files in `bilingual-sentence-pairs` and output cleaned data to `cleaned_data/`.

### 2. Training

To train the model on the cleaned dataset:

```bash
python train.py
```
*Note: The default configuration in `train.py` uses a small model and dataset subset for demonstration purposes.*

### 3. Inference / Demo

To verify the model architecture and see a basic generation loop (with random weights if untrained):

```bash
python main.py
```

## Implementation Details

### Architecture
The model follows the standard Transformer architecture (Vaswani et al., 2017) but adapts it for byte-level processing.
- **Encoder-Decoder**: Standard stack of encoder and decoder layers.
- **Attention**: Scaled Dot-Product Attention with multi-head support.
- **Activation**: GELU activation function.

## License
MIT
