import os
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Ensure NLTK resources are available
nltk.download('stopwords')

# Load the data
def load_data():
    print("Loading data...")
    ham_dir = "dataset/ham"
    spam_dir = "dataset/spam"

    data = []
    
    for file_path in os.listdir(ham_dir):
        with open(os.path.join(ham_dir, file_path), "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            data.append([text, "ham"])
    
    for file_path in os.listdir(spam_dir):
        with open(os.path.join(spam_dir, file_path), "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            data.append([text, "spam"])
    
    data = np.array(data)
    print("Data loaded.")
    return data

# Preprocess data: Remove noise
def preprocess_data(data):
    print("Preprocessing data...")
    punc = string.punctuation
    sw = stopwords.words('english')

    for record in data:
        # Remove punctuation
        record[0] = ''.join([char for char in record[0] if char not in punc])
        # Lowercase and remove stopwords
        words = record[0].split()
        record[0] = ' '.join([word.lower() for word in words if word.lower() not in sw])

    print("Data preprocessed.")
    return data

# Split data into training and test sets
def split_data(data):
    print("Splitting data...")
    features = data[:, 0]  # Email texts
    labels = data[:, 1]    # Labels (ham/spam)
    return train_test_split(features, labels, test_size=0.27, random_state=42)

# Count words in a text
def get_count(text):
    word_counts = {}
    for word in text.split():
        word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts

# Calculate Euclidean distance
def euclidean_difference(count1, count2):
    all_words = set(count1.keys()).union(count2.keys())
    return sum((count1.get(word, 0) - count2.get(word, 0))**2 for word in all_words)**0.5

# Determine class based on K nearest neighbors
def get_class(neighbors):
    spam_count = sum(1 for label, _ in neighbors if label == "spam")
    return "spam" if spam_count > len(neighbors) / 2 else "ham"

# KNN classifier
def knn_classifier(train_data, train_labels, test_data, k):
    print("Running KNN Classifier...")
    results = []
    train_counts = [get_count(text) for text in train_data]

    for test_text in test_data:
        test_count = get_count(test_text)
        distances = [(train_labels[i], euclidean_difference(test_count, train_counts[i])) 
                     for i in range(len(train_data))]
        distances.sort(key=lambda x: x[1])
        neighbors = distances[:k]
        results.append(get_class(neighbors))
    
    return results

# Main program
def main(k):
    data = load_data()
    data = preprocess_data(data)
    train_data, test_data, train_labels, test_labels = split_data(data)

    # Run KNN
    predictions = knn_classifier(train_data, train_labels, test_data, k)

    # Evaluate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Plot accuracy for different K values
    ks = range(1, 20, 2)
    accuracies = []
    for k in ks:
        predictions = knn_classifier(train_data, train_labels, test_data, k)
        accuracies.append(accuracy_score(test_labels, predictions))

    plt.plot(ks, accuracies, marker='o')
    plt.xlabel("K Value")
    plt.ylabel("Accuracy")
    plt.title("KNN Classifier Accuracy")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main(11)
