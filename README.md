# Shakespeare Character-Level Language Model

This project implements two character-level language models trained on the complete works of William Shakespeare. It serves as a pedagogical implementation to understand Transformer architectures and the inner workings of GPT-style models.

## Description

The objective of these models is to predict the next character in a given sequence of text. The project demonstrates the progression from a basic statistical baseline to a state-of-the-art architecture:

1. **Bigram Model**: A baseline implementation using a simple lookup table to map character transition probabilities.
2. **Transformer Model**: A comprehensive implementation of a Transformer decoder, featuring Self-Attention mechanisms, Multi-Head Attention, residual connections, and Layer Normalization.

## Project Structure

* `input.txt`: The training corpus (Shakespeare's collective works).
* `bigram.py`: Training script for the baseline bigram model.
* `transformer.py`: Training script for the optimized Transformer model.
* `requirements.txt`: List of necessary Python dependencies.

## Technical Specifications

The Transformer model utilizes the following hyperparameters:

* **Context Length (Block size)**: 256 characters
* **Embedding Dimension**: 384
* **Transformer Blocks (Layers)**: 6
* **Attention Heads**: 6
* **Optimizer**: AdamW with a learning rate of 3e-4
* **Training Iterations**: 5000

## Usage

To train the model and generate text using the Transformer architecture, run:

```bash
python transformer.py

```

## References

This project was developed following the work of Andrej Karpathy (specifically the "Neural Networks: Zero to Hero" series). The architecture is a simplified implementation of the "Attention Is All You Need" (Vaswani et al., 2017) paper.
