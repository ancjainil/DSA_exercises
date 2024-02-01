# Gutenberg Text Classifier

This Python script demonstrates a text classification pipeline using various machine learning algorithms to classify partitions of Gutenberg texts into different categories.

## Dependencies

- Python 3.x
- `numpy`
- `pandas`
- `nltk`
- `sci-kit-learn`
- `Transformers`
- `torch`
- `matplotlib`

You can install the required dependencies using pip:

pip install numpy pandas nltk scikit-learn transformers torch matplotlib




This Python script performs text classification using various machine learning algorithms on partitions of Gutenberg texts. It includes data preparation, text preprocessing, feature extraction, model training, model evaluation, and subset results analysis.

## 1. Data Preparation
- Randomly selects 6 books from the Gutenberg corpus.
- Divide each book into partitions of text data.

## 2. Text Preprocessing
- Tokenization: Tokenizes the text into words.
- Stopword Removal: Removes stop words from the tokenized text.
- Stemming: Stems words to their base form.
- Lemmatization: Lemmatizes words to their dictionary form.

## 3. Feature Extraction
- TF-IDF Vectorization: Converts text data into numerical features using TF-IDF vectorization.

## 4. Model Training
- Trains various machine learning classifiers, including Naive Bayes, Logistic Regression, SVM, Decision Trees, Random Forest, Gradient Boosting, and KNN.
- Each classifier is trained on the TF-IDF features extracted from the text data.

## 5. Model Evaluation
- Evaluates the performance of each classifier using k-fold cross-validation.
- Calculates metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
- Analyzes bias and variance of each classifier and visualizes them.

## 6. Subset Results
- Prints subset results to compare true and predicted labels for a subset of data.

## Features
- **Text Preprocessing**: Tokenization, stop word removal, stemming, and lemmatization are performed to prepare text data for classification.
- **Feature Extraction**: TF-IDF vectorization converts text data into numerical features.
- **Classification Algorithms**: Evaluate the performance of various classifiers including Naive Bayes, Logistic Regression, SVM, Decision Trees, Random Forest, Gradient Boosting, and KNN.
- **Bias-Variance Analysis**: Calculates and visualizes bias and variance of each classifier to analyze performance.
- **Subset Results**: Prints subset results for comparison between true and predicted labels.

## Usage
1. Open this file in any notebook software such as google collab.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


