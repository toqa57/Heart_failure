import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import class_weight
from sklearn.base import clone
from scipy.stats import zscore
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import warnings

# === Path Configuration ===
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data/heart_failure_clinical_records_dataset.csv"
MODEL_DIR.mkdir(exist_ok=True)


# === Data Preprocessing ===
def preprocess_data():
    df = pd.read_csv(DATA_PATH)

    # Remove numeric outliers using Z-score
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    z_scores = np.abs(zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    return df


# === Training and Calibration ===
def train_and_calibrate_models():
    df = preprocess_data()
    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']

    # Split data: 60% train, 20% calibrate, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_cal_scaled = scaler.transform(X_cal)
    X_test_scaled = scaler.transform(X_test)

    # Class imbalance handling
    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    # === Random Forest ===
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight=class_weights,
        random_state=42
    )
    rf.fit(X_train_scaled, y_train)

    # Calibrate Random Forest
    rf_calibrated = CalibratedClassifierCV(estimator=clone(rf), method='sigmoid')
    rf_calibrated.fit(X_cal_scaled, y_cal)

    rf_acc = rf_calibrated.score(X_test_scaled, y_test)
    print(f"✅ Random Forest Test Accuracy: {rf_acc:.2f}")
    print("\n🔍 Random Forest Test Classification Report:")
    print(classification_report(y_test, rf_calibrated.predict(X_test_scaled), zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, rf_calibrated.predict(X_test_scaled)))

    # === MLP ===
    # Apply SMOTE for both training and calibration
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_scaled, y_train)
    X_cal_bal, y_cal_bal = sm.fit_resample(X_cal_scaled, y_cal)

    mlp = MLPClassifier(
        hidden_layer_sizes=(200, 100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate='adaptive',
        early_stopping=True,
        random_state=42,
        max_iter=300
    )
    mlp.fit(X_train_bal, y_train_bal)

    # Calibrate MLP on balanced calibration set
    mlp_calibrated = CalibratedClassifierCV(estimator=clone(mlp), method='sigmoid')
    mlp_calibrated.fit(X_cal_bal, y_cal_bal)

    mlp_acc = mlp_calibrated.score(X_test_scaled, y_test)
    print(f"✅ MLP Test Accuracy: {mlp_acc:.2f}")
    print("\n🔍 MLP Test Classification Report:")
    print(classification_report(y_test, mlp_calibrated.predict(X_test_scaled), zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, mlp_calibrated.predict(X_test_scaled)))

    # === Save Models and Scaler ===
    joblib.dump(rf_calibrated, MODEL_DIR / "random_forest.pkl")
    joblib.dump(mlp_calibrated, MODEL_DIR / "mlp.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    print("✅ All models and scaler saved in 'models/' folder!")


# === Main Execution ===
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train_and_calibrate_models()