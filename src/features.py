"""Loading and feature engineering for the Kaggle German Credit Risk dataset."""
import numpy as np
import pandas as pd

TARGET = "Risk"                      
SCORE_MAP = {"little": 1, "moderate": 2, "quite rich": 3, "rich": 4}
CATEGORICAL = ["Sex", "Housing", "Purpose"]


def load_data(path):
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])


def engineer(df):
    df = df.copy()
    df["SavingsMissing"] = df["Saving accounts"].isna().astype(int)
    df["CheckingMissing"] = df["Checking account"].isna().astype(int)
    df["SavingsScore"] = df["Saving accounts"].map(SCORE_MAP).fillna(0)
    df["CheckingScore"] = df["Checking account"].map(SCORE_MAP).fillna(0)
    df["TotalLiquidityScore"] = df["SavingsScore"] + df["CheckingScore"]
    df["CreditPerMonth"] = df["Credit amount"] / df["Duration"]
    df["LogCreditAmount"] = np.log1p(df["Credit amount"])
    df["CreditToAge"] = df["Credit amount"] / df["Age"]
    df["LongLoan"] = (df["Duration"] > 24).astype(int)
    df["AgeGroup"] = pd.cut(df["Age"], [0, 25, 35, 45, 60, 120], labels=False)
    return df.drop(columns=["Saving accounts", "Checking account"])


def prepare(path):
    df = engineer(load_data(path))
    y = (df[TARGET] == "bad").astype(int)   # 1 = bad credit risk
    X = df.drop(columns=TARGET)
    return X, y
