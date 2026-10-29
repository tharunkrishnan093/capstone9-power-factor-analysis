from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA = Path("data/simulated_power_factor_readings.csv")
MODEL = Path("models/power_factor_classifier.joblib")
OUT = Path("outputs")
FEATURES = ["voltage_V","current_A","active_power_W","reactive_power_var","load_level","hour","day_of_week"]

def main():
    df = pd.read_csv(DATA)
    before = len(df)
    df = df.drop_duplicates().copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["power_factor"] = df["active_power_W"] / (df["voltage_V"] * df["current_A"])
    df["power_factor"] = df["power_factor"].replace([np.inf,-np.inf], np.nan).clip(0,1)
    df["power_factor"] = df["power_factor"].fillna(df["power_factor"].median())
    df["pf_class"] = np.where(df["power_factor"] < 0.80, "Low",
                       np.where(df["power_factor"] < 0.90, "Moderate", "Good"))
    X, y = df[FEATURES], df["pf_class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=250, max_depth=12,
            min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    OUT.mkdir(exist_ok=True); MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL)
    (OUT/"classification_report.txt").write_text(
        f"Rows before deduplication: {before}\\nRows after deduplication: {len(df)}\\n"
        f"Held-out accuracy: {accuracy_score(y_test,pred):.4f}\\n\\n"
        + classification_report(y_test,pred,zero_division=0), encoding="utf-8")
    pd.DataFrame({"actual":y_test.values,"predicted":pred}).to_csv(OUT/"test_predictions.csv",index=False)
    print((OUT/"classification_report.txt").read_text())
if __name__=="__main__": main()
