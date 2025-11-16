# model_training.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("qd_data.csv")

# Ensure needed columns exist
required = ["material", "band_gap_eV", "radius_nm", "epsilon_r", "crystal_structure"]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# Features & target
X = df[["radius_nm", "epsilon_r"]].copy()
y = df["band_gap_eV"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ML Model
model = RandomForestRegressor(
    n_estimators=600,
    random_state=42
)

model.fit(X_train, y_train)

print("Training done!")
print("R^2 score:", model.score(X_test, y_test))

# Save model
joblib.dump(model, "qd_model.joblib")
print("Saved as qd_model.joblib")
