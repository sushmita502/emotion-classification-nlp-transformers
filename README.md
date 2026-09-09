Emotion Classification from Text

This project explores how different NLP models perform when the goal is
to understand the emotion expressed in a short piece of text.

The project compares three approaches: a fully connected neural network
using TF-IDF features, a recurrent neural network, and a pretrained
Transformer model. The final Transformer model uses DistilBERT and
is fine-tuned for six emotion classes: anger, fear, joy, love,
sadness, and surprise.

Project Overview

The main idea was not only to train a model, but also to compare
different ways of representing and understanding text.

The project follows this workflow:

Load and inspect the emotion dataset

Explore the distribution of the emotion classes

Clean the text

Build a TF-IDF based neural network as a baseline

Experiment with a recurrent neural network

Fine-tune DistilBERT for emotion classification

Evaluate the models using classification metrics and confusion
matrices

Compare the strengths and limitations of the different approaches

Dataset

The dataset contains short text samples paired with an emotion label.

Training samples: 15,999

Test samples: 1,999

Number of emotion classes: 6

The classes are:

Anger

Fear

Joy

Love

Sadness

Surprise

The dataset is not evenly distributed. For example, the training set
contains many more examples of joy and sadness than
surprise. This class imbalance is important when interpreting the
model's performance.

Text Preprocessing

Before training the models, basic cleaning was applied to the text:

Converted text to lowercase

Removed URLs

Removed non-alphabetic characters

Kept the cleaned text for the different modelling approaches

Models

1. TF-IDF + Fully Connected Neural Network

The first model is used as a baseline.

The text is converted into TF-IDF features and passed through a fully
connected neural network with dense layers and dropout.

This approach performs reasonably well when emotion can be identified
from individual words, but it has a limitation: TF-IDF does not preserve
word order or deeper contextual relationships.

2. Recurrent Neural Network

A recurrent approach was explored to work with the sequential nature of
text.

Unlike a simple bag-of-words representation, recurrent models can use
information from the sequence of words when making a prediction.

3. DistilBERT Transformer

The final approach uses DistilBERT, a lightweight pretrained
Transformer model.

The pretrained model is fine-tuned for the six emotion classes using the
Hugging Face Transformers library. This allows the model to make use of
contextual representations learned from large-scale text data.

The Transformer performed better than the simpler neural-network
approaches in this project.

Results

The DistilBERT model achieved 92% accuracy on the 2,000-sample
evaluation output recorded in the notebook.

Its per-class F1 scores were:

Emotion      Precision   Recall   F1-score

Anger             0.92     0.92       0.92
Fear              0.86     0.91       0.88
Joy               0.95     0.94       0.94
Love              0.82     0.82       0.82
Sadness           0.96     0.96       0.96
Surprise          0.80     0.67       0.73

The results show that the model performs particularly well on
sadness and joy. Performance is lower for surprise, which is
also the least represented class in the training data.

What I Learned

This project helped me understand the impact of text representation and
model architecture on NLP classification tasks.

The main takeaway was that traditional feature-based models can provide
a useful baseline, but pretrained Transformers are much better at
capturing context and subtle semantic differences between emotions.

The project also highlighted the importance of looking beyond overall
accuracy. Because the dataset is imbalanced, class-level precision,
recall, and F1-score give a clearer picture of where the model performs
well and where it still struggles.

Technologies

Python

Pandas

NumPy

Scikit-learn

TensorFlow / Keras

PyTorch

Hugging Face Transformers

DistilBERT

Matplotlib

Repository Structure

emotion-classification-nlp-transformers/
│
├── notebooks/
│   └── Emotion_Classification_NLP.ipynb
│
├── README.md
└── requirements.txt

How to Run

The notebook was developed using Google Colab and uses GPU acceleration
for the Transformer model.

Clone the repository.

Install the required Python packages.

Open the notebook in Jupyter or Google Colab.

Update the dataset path to match your local environment.

Run the notebook cells in order.

The dataset itself is not included in this repository.

Limitations and Possible Improvements

There are a few areas that could be improved in a future version:

Address the class imbalance, especially for the surprise class.

Tune the Transformer hyperparameters more systematically.

Compare additional pretrained language models.

Add a simple inference interface where a user can enter text and
receive the predicted emotion.

Evaluate the models on additional unseen text.

Project Status

Completed as an NLP/deep learning project, with the main focus on
comparing traditional neural approaches with a pretrained Transformer
for emotion classification.
