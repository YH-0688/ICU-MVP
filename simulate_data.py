import pandas as pd, numpy as np, datetime as dt, random
np.random.seed(42)

BEDS = 15
HOURS = 24
now = dt.datetime.now()

rows = []
for bed in range(1, BEDS + 1):
    map_base = random.randint(65, 85)           # mean arterial pressure
    hr_base  = random.randint(70, 100)          # heart rate
    spo2_base = random.randint(92, 98)

    for h in range(HOURS):
        t = now - dt.timedelta(hours=(HOURS - h))
        rows.append({
            "timestamp" : t,
            "bed"       : f"Bed {bed}",
            "MAP"       : max(40, np.random.normal(map_base, 8)),
            "HR"        : max(40, np.random.normal(hr_base, 10)),
            "SpO2"      : max(70, np.random.normal(spo2_base, 3)),
            "Lactate"   : abs(np.random.normal(1.8, 0.6)),
            "Creat"     : abs(np.random.normal(1.2, 0.3)),
            "Pressor"   : np.random.binomial(1, 0.25)
        })
df = pd.DataFrame(rows)
df.to_csv("icu_raw.csv", index=False)
print("✓  generated icu_raw.csv")
