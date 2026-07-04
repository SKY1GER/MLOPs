import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# MLflow Tracking Server
mlflow.set_tracking_uri("http://localhost:7004")

# Experiment Name
mlflow.set_experiment("Iris-Classification")

# Load Dataset
df = pd.read_csv("iris.csv")

# Features and Target
X = df.drop("species", axis=1)
y = df["species"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

with mlflow.start_run():

    # Parameters
    n_estimators = 100
    random_state = 42

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Log Parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)

    # Log Metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Save Model Locally
    joblib.dump(model, "iris_model.pkl")

    # Log Model Artifact
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    # Log Additional Artifact
    mlflow.log_artifact("iris_model.pkl")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("Model logged to MLflow successfully!")
