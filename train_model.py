import pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

df = pd.read_csv("icu_raw.csv")

# naïve label: any data point where MAP < 60 **or** Lactate > 2.5 counts as “deterioration soon”
df["label"] = ((df["MAP"] < 60) | (df["Lactate"] > 2.5)).astype(int)

X = df[["MAP","HR","SpO2","Lactate","Creat","Pressor"]]
y = df["label"]

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    LogisticRegression())
]).fit(X, y)

joblib.dump(pipe, "model.pkl")
print("✓  trained model (AUROC = %.2f)" % roc_auc_score(y, pipe.predict_proba(X)[:,1]))
