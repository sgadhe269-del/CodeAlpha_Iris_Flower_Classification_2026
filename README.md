# CodeAlpha — Iris Flower Classification

## Objective
Train a machine-learning classifier to predict Iris flower species (setosa, versicolor, virginica) from sepal/petal measurements.

## Tools
Python, pandas, scikit-learn, matplotlib, joblib.

## Run in VS Code
1. Open this folder in VS Code.
2. Create and activate a virtual environment (instructions in the parent folder's `SETUP_GUIDE.md`).
3. Install dependencies: `python -m pip install -r requirements.txt`
4. Run: `python main.py`

The Iris dataset is included in scikit-learn, so a separate dataset download is not needed.

## Workflow
- Load and inspect data
- Check missing values and class counts
- Split data into train/test sets
- Standardize features and train Logistic Regression
- Evaluate accuracy and classification metrics
- Save a confusion-matrix image and trained model

## Outputs
- `outputs/confusion_matrix.png`
- `models/iris_model.joblib`

## Important
The test accuracy can vary if you change the split or model. Report the accuracy printed by your own run; do not invent results.
