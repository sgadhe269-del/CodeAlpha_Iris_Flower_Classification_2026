"""
CodeAlpha Internship — Task 1: Iris Flower Classification
Run: python main.py
"""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

def main():
    # 1. Load the built-in Iris dataset (no manual download required).
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    df = X.copy()
    df["species"] = y.map(dict(enumerate(iris.target_names)))

    print("\n=== DATA PREVIEW ===")
    print(df.head())
    print("\n=== DATASET SHAPE ===")
    print(f"Rows: {df.shape[0]}, columns: {df.shape[1]}")
    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())
    print("\n=== SPECIES COUNTS ===")
    print(df["species"].value_counts())

    # 2. Split data before training to evaluate on unseen examples.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Scale measurements and train a classification model.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(X_train, y_train)

    # 4. Evaluate.
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n=== TEST ACCURACY ===\n{accuracy:.2%}")
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, predictions, target_names=iris.target_names))

    # 5. Save confusion matrix and trained model.
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=iris.target_names, cmap="Blues"
    )
    plt.title("Iris Classification — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    joblib.dump(
        {"model": model, "feature_names": list(X.columns), "target_names": list(iris.target_names)},
        MODEL_DIR / "iris_model.joblib"
    )

    # 6. Example prediction using measurements in centimeters.
    sample = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=X.columns
    )
    predicted_class = model.predict(sample)[0]
    print("\n=== EXAMPLE PREDICTION ===")
    print(f"Measurements: {sample.iloc[0].to_dict()}")
    print(f"Predicted species: {iris.target_names[predicted_class]}")
    print(f"\nSaved model to: {MODEL_DIR / 'iris_model.joblib'}")
    print(f"Saved chart to: {OUTPUT_DIR / 'confusion_matrix.png'}")

if __name__ == "__main__":
    main()
