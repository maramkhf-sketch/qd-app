# qd_pipeline.py — Hybrid ML + Physics Doping Model
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

class QDSystem:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.pipeline = None

    # ---------------------- TRAIN ----------------------
    def fit(self, df: pd.DataFrame):
        df = df.copy()

        X = df[["material", "radius_nm", "epsilon_r", "crystal_structure"]]
        y = df["band_gap_eV"]

        cat_cols = ["material", "crystal_structure"]
        num_cols = ["radius_nm", "epsilon_r"]

        pre = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols),
                ("num", "passthrough", num_cols),
            ]
        )

        self.pipeline = Pipeline(
            steps=[
                ("pre", pre),
                ("model", RandomForestRegressor(n_estimators=600, random_state=42)),
            ]
        )

        self.pipeline.fit(X, y)

    # ---------------------- BASE PREDICTION ----------------------
    def predict_base(self, material, radius_nm, epsilon_r, crystal_structure):
        X = pd.DataFrame([{
            "material": material,
            "radius_nm": float(radius_nm),
            "epsilon_r": float(epsilon_r),
            "crystal_structure": crystal_structure
        }])
        return float(self.pipeline.predict(X)[0])

    # ---------------------- DOPING PHYSICS MODEL ----------------------
    def doping_shift(self, dopant, doping_type, conc):
        """
        Adds physical correction to band gap.
        conc is in cm^-3.
        """

        if not dopant or not doping_type or conc is None or conc <= 0:
            return 0.0

        dopant = dopant.capitalize()

        # Typical dopant band edge effects (eV)
        dopant_strength = {
            "B": -0.12, "N": +0.18, "P": +0.15, "As": +0.22,
            "Al": +0.05, "Ga": +0.07, "In": +0.03,
            "Mn": -0.25, "Cu": -0.18, "Fe": -0.30,
            "Cl": +0.10, "Br": +0.12, "I": +0.15,
        }

        base = dopant_strength.get(dopant, 0.0)

        # n-type increases conduction band, p-type decreases valence band
        if doping_type == "n":
            direction = +1
        else:
            direction = -1

        # Normalize concentration effect
        factor = np.log10(conc) - 17
        factor = max(min(factor, 3), -3)  # clamp

        return base * direction * (factor * 0.25)

    # ---------------------- FINAL PREDICTION ----------------------
    def predict_bandgap(self, material, radius_nm, epsilon_r, crystal_structure,
                        dopant=None, doping_type=None, doping_conc_cm3=None):

        Eg0 = self.predict_base(material, radius_nm, epsilon_r, crystal_structure)

        shift = self.doping_shift(dopant, doping_type, doping_conc_cm3)
        return float(Eg0 + shift)

    # ---------------------- INVERSE SEARCH ----------------------
    def suggest_materials_from_bandgap(self, target, radius_nm, epsilon_r,
                                       crystal_structure, top_k=3):

        # We search inside the known dataset only
        df = pd.read_csv("qd_data.csv")
        df["pred"] = df.apply(
            lambda r: self.predict_base(r["material"], r["radius_nm"],
                                        r["epsilon_r"], r["crystal_structure"]),
            axis=1
        )

        df["score"] = 1 / (1 + abs(df["pred"] - target))
        top = df.sort_values("score", ascending=False).head(top_k)

        return [(row.material, float(row.score)) for _, row in top.iterrows()], top
