import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py shopping.csv")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(np.array(y_test) == np.array(predictions)).sum()}")
    print(f"Incorrect: {(np.array(y_test) != np.array(predictions)).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).
    """
    # Define month mapping
    abbr_to_num = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5, 'Jul': 6,
                   'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}
    
    # Read data with pandas
    df = pd.read_csv(filename)
    
    # Convert categorical values to integers
    df["Month"]       = df["Month"].map(abbr_to_num)
    df["VisitorType"] = (df["VisitorType"] == "Returning_Visitor").astype(int)
    df["Weekend"]     = (df["Weekend"] == "TRUE").astype(int)
    df["Revenue"]     = (df["Revenue"] == "TRUE").astype(int)
    
    # Separate features and labels
    evidence = df.iloc[:, :-1].values.tolist()
    labels   = df.iloc[:, -1].values.tolist()
    
    return evidence, labels

def train_model(evidence, labels):
    """
    Train a k-nearest neighbor classifier (k=1) on the training data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model

def evaluate(labels, predictions):
    """
    Calculate sensitivity and specificity of the model.
    """
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    true_positives = np.sum((labels == 1) & (predictions == 1))
    false_negatives = np.sum((labels == 1) & (predictions == 0))
    true_negatives = np.sum((labels == 0) & (predictions == 0))
    false_positives = np.sum((labels == 0) & (predictions == 1))
    
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    
    return sensitivity, specificity

if __name__ == "__main__":
    main()
