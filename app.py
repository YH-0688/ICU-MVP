import streamlit as st
import pandas as pd
import joblib
import pathlib
import time
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ICU Early-Warning MVP", layout="wide")

# ---------- data & model ----------
if not pathlib.Path("icu_raw.csv").exists():
    import simulate_data  # on first run this creates icu_raw.csv

df = pd.read_csv("icu_raw.csv")

if not pathlib.Path("model.pkl").exists():
    import train_model  # on first run this trains & saves model.pkl

model: joblib = joblib.load("model.pkl")
# Compute risk score (probability) per row
df["risk"] = model.predict_proba(
    df[["MAP", "HR", "SpO2", "Lactate", "Creat", "Pressor"]]
)[:, 1]

# ---------- UI ----------
st.title("🛎️  ICU Early-Warning Prototype — VitaBoard v0.1")

# Show latest record per bed, sorted by risk descending
latest = (
    df.sort_values("timestamp")
    .groupby("bed")
    .tail(1)
    .set_index("bed")
    .sort_values("risk", ascending=False)
)

def colour(r):
    if r >= 0.7:
        return "🔥 High"
    if r >= 0.4:
        return "⚠️  Medium"
    return "🟢 Low"

show = latest[["MAP", "HR", "SpO2", "Lactate", "risk"]].rename(
    columns={"risk": "Risk (0–1)"}
)
show["Risk Level"] = show["Risk (0–1)"].apply(colour)
st.subheader("Bed risk ranking (auto-refresh every 30 s)")
st.table(
    show.style.format(
        {"MAP": "{:.0f}", "HR": "{:.0f}", "SpO2": "{:.0f}", "Lactate": "{:.2f}", "Risk (0–1)": "{:.2f}"}
    )
)

# Dropdown to pick a bed
bed = st.selectbox("🔍  Investigate bed", options=latest.index)
if bed:
    row = latest.loc[bed]
    st.markdown(f"### Why is **{bed}** at risk?")
    # Extract feature values for this bed
    X_vals = np.array(
        [row["MAP"], row["HR"], row["SpO2"], row["Lactate"], row["Creat"], row["Pressor"]]
    ).reshape(1, -1)

    # Extract model pipeline: scaler + logistic
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]

    # Scale features and get coefficients
    X_scaled = scaler.transform(X_vals)
    coefs = clf.coef_.flatten()  # array of length 6
    intercept = clf.intercept_[0]

    # Calculate “contribution” = coef_i * x_scaled_i
    contribs = coefs * X_scaled.flatten()
    features = ["MAP", "HR", "SpO2", "Lactate", "Creat", "Pressor"]

    # Build a bar chart of contributions
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(features, contribs, color="tab:blue")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Scaled-Feature × Coefficient")
    ax.set_title("Feature contributions to log-odds")
    st.pyplot(fig)

    # Show numeric breakdown below
    contrib_df = pd.DataFrame({
        "Feature": features,
        "Value": X_vals.flatten(),
        "Scaled value": X_scaled.flatten(),
        "Coefficient": coefs,
        "Contribution": contribs,
    })
    st.dataframe(contrib_df.style.format({
        "Value": "{:.2f}",
        "Scaled value": "{:.2f}",
        "Coefficient": "{:.2f}",
        "Contribution": "{:.2f}"
    }))

st.caption("Prototype only – synthetic data – no patient identifiers.")

# auto-refresh every 30 s
time.sleep(30)
st.experimental_rerun()
