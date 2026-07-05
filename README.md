# Credit Scoring Model

A professional machine learning project for predicting whether a customer is likely to be a good credit risk. The project generates synthetic financial data, engineers meaningful features, splits the data into training and testing sets, trains multiple classification models, and evaluates their performance.

## Project Overview

This repository demonstrates a complete end-to-end machine learning workflow for a credit scoring use case:

- Synthetic dataset generation
- Feature engineering
- Train/test data splitting
- Model training and evaluation
- Model persistence for later use

## Project Structure

- `train.py` — generates data, creates train/test files, and trains the models
- `test.py` — evaluates the saved models on the test data
- `data/` — contains the generated datasets
- `models/` — stores trained model files
- `requirements.txt` — Python dependencies required to run the project

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Train the models

```bash
python train.py
```

### Evaluate the models

```bash
python test.py
```

## Models Included

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

## Notes

The dataset used in this project is synthetic and intended for educational and demonstration purposes.
