
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "data" / "student_performance.csv")

features = ["Study_Hours", "Attendance", "Internal_Marks", "Assignment_Marks", "Previous_Performance"]
X = df[features]
y = (df["Result"] == "Pass").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=250, max_depth=8, random_state=42, class_weight="balanced"
    ))
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
metrics = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred, zero_division=0),
    "recall": recall_score(y_test, pred, zero_division=0),
    "f1": f1_score(y_test, pred, zero_division=0),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist()
}

joblib.dump({"model": model, "features": features, "metrics": metrics}, BASE / "models" / "student_model.pkl")
print("Model trained successfully.")
print(metrics)
