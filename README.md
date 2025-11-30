# NumPy Byte-Level Transformer

A fully functional Transformer model implemented from scratch using only **NumPy**. This project demonstrates the inner workings of the Transformer architecture, including self-attention, backpropagation, and optimization, without relying on deep learning frameworks like PyTorch or TensorFlow.

## Features

*   **Pure NumPy Implementation**: No automatic differentiation engines. Every forward and backward pass is manually implemented.
*   **Byte-Level Tokenization**: Operates directly on UTF-8 bytes (vocab size 260), allowing it to process any text string without a pre-computed vocabulary.
*   **Custom Autograd**: `backward()` methods implemented for all layers (`Linear`, `LayerNorm`, `GELU`, `Embedding`, `MultiHeadAttention`, `Transformer`).
*   **Optimization**: Custom `Adam` optimizer implementation.
*   **Training Loop**: Complete training pipeline with `CrossEntropyLoss`.

## Project Structure

*   `config.py`: Configuration dataclass for model hyperparameters.
*   `layers.py`: Core neural network layers (`Linear`, `LayerNorm`, `GELU`, `Softmax`, `Embedding`, `PositionalEncoding`) with forward and backward passes.
*   `attention.py`: Multi-Head Attention mechanism implementation.
*   `transformer.py`: Full Transformer architecture (`Encoder`, `Decoder`, `Transformer`).
*   `optimizer.py`: `Adam` optimizer implementation.
*   `loss.py`: `CrossEntropyLoss` implementation.
*   `utils.py`: Helper functions for tokenization and masking.
*   `clean_dataset.py`: Script to clean and prepare the bilingual dataset.
*   `train.py`: Main training script.
*   `main.py`: Demo script for inference and shape verification.

## Requirements

*   Python 3.x
*   NumPy

```bash
pip install numpy
```

## Usage

### 1. Data Preparation

The project expects a bilingual dataset. If you have the raw data (e.g., from Tatoeba), run the cleaning script to format it:

```bash
python clean_dataset.py
```
This will process files in `bilingual-sentence-pairs/versions/3` and output cleaned data to `cleaned_data/`.

### 2. Training

To train the model on the cleaned dataset:

```bash
python train.py
```
*Note: The default configuration in `train.py` uses a small model and dataset subset for demonstration purposes, as training on CPU with pure NumPy is computationally intensive.*

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

### Backward Pass
Unlike frameworks that build a computation graph, this project manually calculates gradients for every operation. Each module has a `backward(grad)` method that:
1.  Computes gradients with respect to its inputs (`dL/dx`).
2.  Computes gradients with respect to its weights (`dL/dW`).
3.  Returns `dL/dx` to be passed to the previous layer.

### Optimization
The `Adam` optimizer tracks first and second moments of the gradients to update parameters adaptively.

## License
MIT
