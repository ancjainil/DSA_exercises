import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import gutenberg
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

class GutenbergPartitioner:
    def __init__(self , genre='fiction'):
        # Download the books
        nltk.download('gutenberg')

        # Select a random book
         # Select books from a specific genre (e.g., fiction)
        self.all_books = gutenberg.fileids()
        self.selected_books = random.sample(self.all_books, 6)  # Choose 6 books from the same genre
        self.book_data = [(book, gutenberg.raw(book)) for book in self.selected_books]

    def split_into_partitions(self, num_partitions=200, partition_size=100):
        partitions = []
        for book, data in self.book_data:
            words = word_tokenize(data)
            for _ in range(num_partitions):
                start_index = random.randint(0, len(words) - partition_size)
                partition = ' '.join(words[start_index:start_index + partition_size])
                partitions.append((book, partition))

        # Shuffle the order of partitions
        random.shuffle(partitions)

        # Label the partitions with alphabetical labels
        labels = {book: chr(ord('a') + i) for i, book in enumerate(self.selected_books)}

        # Add the labels to the partitions
        labeled_partitions = [(labels[book], partition) for book, partition in partitions]
        return labeled_partitions

    def save_to_csv(self, labeled_partitions, output_filename='random_partitions.csv'):
        # Serialize the data using Pandas
        df = pd.DataFrame(labeled_partitions, columns=['Book', 'Partition'])
        df.to_csv(output_filename, index=False)

def preprocess_text(text):
    # Tokenize, remove stopwords, and non-alphabetic characters
    tokens = nltk.word_tokenize(text)
    stopwords = set(nltk.corpus.stopwords.words('english'))
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stopwords]
    return ' '.join(tokens)

def main():
    partitioner = GutenbergPartitioner()
    labeled_partitions = partitioner.split_into_partitions()
    partitioner.save_to_csv(labeled_partitions)

    # Load labeled partitions from CSV
    df = pd.read_csv('random_partitions.csv')

    # Preprocess the data
    df['Processed_Partition'] = df['Partition'].apply(preprocess_text)

    # Feature extraction using TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    features = vectorizer.fit_transform(df['Processed_Partition']).toarray()
    labels = df['Book']

    # Train a machine learning model (Naive Bayes)
    classifier = MultinomialNB()
    scores = cross_val_score(classifier, features, labels, cv=10)
    mean_accuracy = scores.mean()
    print(f'Mean Accuracy: {mean_accuracy}')

    # Error analysis
    classifier.fit(features, labels)
    predictions = classifier.predict(features)

    # Confusion matrix and classification report
    cm = confusion_matrix(labels, predictions)
    cr = classification_report(labels, predictions)

    print('Confusion Matrix:')
    print(cm)
    print('Classification Report:')
    print(cr)

    # Visualize confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == "__main__":
    main()
