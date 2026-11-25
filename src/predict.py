import joblib
import pandas as pd
from features import engineer

model = joblib.load("outputs/model.joblib")
new = pd.read_csv("data/new_applicants.csv")

proba = model.predict_proba(engineer(new))[:, 1]
new["BadRiskProbability"] = proba.round(3)
new["Prediction"] = ["bad" if p >= 0.5 else "good" for p in proba]
print(new[["Age", "Credit amount", "Duration", "BadRiskProbability", "Prediction"]])