import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


df = pd.read_csv("dataset/winequality-red-selected-missing.csv")
print(df.info())

df['quality'] = df['quality'].apply(lambda x: 1 if x > 6 else 0)
print(df.head())

X = df.drop('quality', axis=1)
y = df['quality']

imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)

X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed_df)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42
)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

def evaluate_model(model, threshold=0.6):
    model.fit(X_train_bal, y_train_bal)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print("\n===== Random Forest =====")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Good", "Good"]))

    precision = precision_score(y_test, y_pred)
    print(f"Precision (Good Wines): {precision:.4f}")

    return model, precision

rf_model = RandomForestClassifier(random_state=42)
best_model, best_precision = evaluate_model(rf_model)

try:
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    print("Model and preprocessors saved in 'models/' directory.")
except Exception as e:
    print("Error occurred during saving:", e)
