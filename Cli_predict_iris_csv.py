import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def train_model_from_csv(csv_path='iris.csv'):
    """Trains model using data from CSV file."""
    df = pd.read_csv(csv_path)

    
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    if not all(col in df.columns for col in feature_cols):
        feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        target_col = 'Species'
    else:
        target_col = 'species'

    X = df[feature_cols]
    y = df[target_col]
    
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)
    
    return clf, clf.classes_

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Iris Species (CSV Backend)")
    parser.add_argument('--sl', type=float, help='Sepal Length')
    parser.add_argument('--sw', type=float, help='Sepal Width')
    parser.add_argument('--pl', type=float, help='Petal Length')
    parser.add_argument('--pw', type=float, help='Petal Width')
    parser.add_argument('--csv', type=str, default='iris.csv', help='Path to training CSV')

    args = parser.parse_args()

    # Get inputs
    if not all([args.sl, args.sw, args.pl, args.pw]):
        print("--- Enter Flower Dimensions ---")
        sl = float(input("Sepal Length: "))
        sw = float(input("Sepal Width:  "))
        pl = float(input("Petal Length: "))
        pw = float(input("Petal Width:  "))
    else:
        sl, sw, pl, pw = args.sl, args.sw, args.pl, args.pw

    # Train and Predict
    clf, classes = train_model_from_csv(args.csv)
    input_data = pd.DataFrame([[sl, sw, pl, pw]], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    
    
    prediction = clf.predict(input_data.values)[0]
    prob = np.max(clf.predict_proba(input_data.values))

    print(f"\nPrediction: {prediction}")
    print(f"Confidence: {prob:.2%}")