# Name Generator

## Overview

This project is a small character-level name generator model. It learns which letters commonly follow one another in a dataset of 32,033 names, then samples those learned patterns to create new names through two model

1. bigram-count-matrix or table and its multinomial predictions
2. neural network bigram trained simple model

## How It Works

It surrounds every name with a special `.` start/end token and counts each adjacent character pair (bigram). These counts form a 27 × 27 matrix for all unique characters in the names then sorted plus the boundary token. Add-one smoothing prevents zero probabilities, and PyTorch's multinomial sampler generates names one character at a time. Negative log-likelihood is used to evaluate model quality.

The final section also demonstrates an equivalent single-layer neural network using one-hot inputs, softmax probabilities, gradient descent, and L2 regularization. This teaching example currently trains on only the first name.

## Files

- `biagram.ipynb` — data exploration, modeling, evaluation, and sampling
- `names.txt` — training dataset
- `bigram_counter_matrix_names_data.png` — transition-count visualization

## Run

The whole project is setup in poetry : read the pyproject.toml for required information.

```bash
# make poetry is installed in your device and then run

 poetry lock && poetry install

```

Using the fixed random seeds makes generated samples reproducible.

## Thank you

[Andrej Karpathy](https://karpathy.ai/)
