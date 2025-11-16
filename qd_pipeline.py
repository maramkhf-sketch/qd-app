# qd_pipeline.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

class QDSystem:
    def __init__(self):
        self.material_df = pd.read_csv("qd_data.csv")
        self.model = None
        self._train_ml_model()

    # ------------------------------
    # 1) Train ML correction model
    # ------------------------------
    def _train_ml_model(self):
        df = self.material_df.copy()

        df = df.dropna(subset=["radius_nm", "epsilon_r", "band_gap_eV"])

        X = df[["radius_nm", "epsilon_r"]]
        y = df["band_gap_eV"]

        model = RandomForestRegressor(
            n_estimators=120,
            max_depth=8,
            random_state=42
        )
        model.fit(X, y)

        self.model = model

    # -----------------------------------
    # 2) Brus equation (base physics)
    # -----------------------------------
    def brus_equation(self, radius_nm, eps_r):
        if radius_nm <= 0:
            return np.nan

        R = radius_nm * 1e-9
        eps0 = 8.854e-12
        hbar = 1.055e-34
        me = 9.11e-31
        e = 1.6e-19

        Eg_bulk = 1.5  # default; overridden by CSV
        if "Eg_bulk_eV" in self.material_df.columns:
            pass

        K = (374 * (1 / radius_nm**2))  # nanoscale approximation

        return Eg_bulk + K

    # -----------------------------------
    # 3) ML correction (B2)
    # -----------------------------------
    def ml_correction(self, radius_nm, eps_r):
        base_pred = self.model.predict([[radius_nm, eps_r]])[0]
        brus_base = 1.5 + 374 * (1 / radius_nm**2)

        correction = (base_pred - brus_base) * 0.35  # B2 strength

        return correction

    # -----------------------------------
    # 4) Forward Prediction
    # -----------------------------------
    def predict(self, material, radius_nm, eps_r, doping_conc=0):
        base = 1.5 + 374 * (1 / radius_nm**2)
        corr = self.ml_correction(radius_nm, eps_r)

        dop = 0
        if doping_conc > 0:
            dop = 0.02 * np.log10(doping_conc)

        Eg = base + corr + dop
        return round(Eg, 4)

    # -----------------------------------
    # 5) Curve data
    # -----------------------------------
    def curve(self, eps_r):
        radii = np.linspace(1, 10, 50)
        Eg_list = []

        for r in radii:
            base = 1.5 + 374 * (1 / r**2)
            corr = self.ml_correction(r, eps_r)
            Eg_list.append(base + corr)

        return radii, Eg_list
