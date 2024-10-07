# Text Classification using Logistic Regression

This project uses logistic regression to classify text data. The dataset includes text samples and their corresponding labels. The text data is preprocessed using NLTK, and TF-IDF is used for feature extraction.

## Requirements

- pandas
- nltk
- scikit-learn

## Dataset

The dataset used is `dataset.csv`, which contains text samples and their corresponding labels.

## Steps

1. **Data Preparation**: Load the dataset and preprocess the text data by converting it to lowercase, tokenizing, removing stopwords, and stemming.
2. **Feature Extraction**: Extract features from the text using TF-IDF.
3. **Model Training**: Train a logistic regression model to classify the text data.
4. **Evaluation**: Evaluate the model's performance using accuracy, precision, recall, F1 score, and confusion matrix.

### Data Preparation

Load the dataset and preprocess the text data:

```python
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("dataset.csv")

# Define preprocessing function
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return ' '.join(stemmed_tokens)

# Apply preprocessing to 'text_' column
df['preprocessed_text'] = df['text_'].apply(preprocess_text)
