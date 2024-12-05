# Sentiment Analysis with LSTM

This project aims to automate sentiment analysis using a Long Short-Term Memory (LSTM) network. The model classifies movie reviews from the IMDB dataset into positive or negative sentiment using deep learning techniques, specifically an LSTM network with embeddings.

## Project Overview

This project utilizes the IMDB dataset, which contains 50,000 movie reviews labeled as either positive or negative. The goal is to build an LSTM-based model that can automatically classify the sentiment of the reviews.

### Steps Involved:
1. **Data Preprocessing:**
   - Download the IMDB dataset.
   - Clean and tokenize the text data.
   - Apply padding to ensure uniform input size.

2. **Word Embedding:**
   - Train a Word2Vec model on the training set.
   - Create an embedding matrix from the Word2Vec model.

3. **Model Architecture:**
   - Build a Sequential model with an Embedding layer, a Bidirectional LSTM layer, and a Dense layer for classification.
   - Use regularization techniques such as L2 regularization and Dropout to avoid overfitting.

4. **Training and Evaluation:**
   - Train the model using the Adam optimizer and binary cross-entropy loss function.
   - Monitor training performance with EarlyStopping and save the best model using ModelCheckpoint.

5. **Model Evaluation:**
   - Evaluate the model's performance on the test dataset.
   - Display the confusion matrix and classification report.
   - Plot training and validation loss and accuracy.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- scikit-learn
- pandas
- numpy
- gensim
- matplotlib
- seaborn

## Installation

You can install the necessary libraries using pip:

```bash
pip install tensorflow scikit-learn pandas numpy gensim matplotlib seaborn
```

## Data

The project uses the IMDB dataset, which can be accessed via the `aclImdb_v1.tar.gz` file. This dataset includes 50,000 movie reviews, with an equal distribution of positive and negative labels.

## Usage

### Data Loading

The data is loaded from the IMDB dataset, which is first unzipped and extracted. Reviews are read from the 'neg' and 'pos' directories, and labels are assigned accordingly (0 for negative, 1 for positive).

### Results

- **Accuracy:** 88.72% on the test data.
- **Confusion Matrix & Classification Report:** Visualized with seaborn heatmap.
- **Loss and Accuracy Plots:** Visualized over training epochs.
