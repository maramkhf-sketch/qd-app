# qd_pipeline.py — Clean Version (NO pymatgen)

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor

class QDSystem:
    """
    Hybrid ML + physics model
    - Learns from the CSV dataset (materials & bandgaps)
    - Predicts band gap for unseen materials using nearest neighbors
    - Supports doping additively (optional)
    """

    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.nn = None
        self.df = None

    # -------------------------------------------------------
    #                TRAIN MODEL ON CSV DATA
    # -------------------------------------------------------
    def fit(self, df: pd.DataFrame):
        self.df = df.copy()

        # Basic cleaned columns
        X = df[["radius_nm", "epsilon_r"]].copy()
        mat = df["material"].values.reshape(-1, 1)

        # Encode materials
        mat_enc = self.encoder.fit_transform(mat)

        # Final ML input
        X_full = np.hstack([X.values, mat_enc])
        y = df["band_gap_eV"].values

        # Train regressor
        self.model.fit(X_full, y)

        # Build nearest-neighbors model for fallback
        self.nn = NearestNeighbors(n_neighbors=3)
        self.nn.fit(X_full)

    # -------------------------------------------------------
    #                INTERNAL ENCODING HELPER
    # -------------------------------------------------------
    def encode_input(self, material, radius_nm, epsilon_r):
        """Encode inputs for the RF model."""
        Xnum = np.array([[radius_nm, epsilon_r]])
        Xmat = self.encoder.transform([[material]])  # (1, n_encoded)
        return np.hstack([Xnum, Xmat])

    # -------------------------------------------------------
    #            PREDICT BAND GAP (MAIN FORWARD)
    # -------------------------------------------------------
    def predict_bandgap(self, material, radius_nm, epsilon_r, crystal_structure=None,
                        dopant=None, doping_type=None, doping_conc=None):

        # 1) Encode input
        try:
            X = self.encode_input(material, radius_nm, epsilon_r)
            base_pred = float(self.model.predict(X)[0])
        except Exception:
            # Material not seen -> fallback to NN on radius + epsilon only
            base_pred = self._predict_with_neighbors(radius_nm, epsilon_r)

        # 2) Apply doping (optional)
        doped_pred = self._apply_doping(base_pred, dopant, doping_type, doping_conc)

        return doped_pred

    # -------------------------------------------------------
    #       NN fallback method (for unknown materials)
    # -------------------------------------------------------
    def _predict_with_neighbors(self, radius_nm, epsilon_r):
        """Fallback when material not in training set."""
        # Just find NN using radius & epsilon only
        df = self.df.copy()
        df["dr"] = (df["radius_nm"] - radius_nm) ** 2
        df["de"] = (df["epsilon_r"] - epsilon_r) ** 2
        df["dist"] = np.sqrt(df["dr"] + df["de"])

        nearest = df.nsmallest(3, "dist")
        return float(nearest["band_gap_eV"].mean())

    # -------------------------------------------------------
    #             PHYSICS-INSPIRED DOPING EFFECT
    # -------------------------------------------------------
    def _apply_doping(self, Eg, dopant, doping_type, conc):

        # No doping provided → return unchanged
        if not dopant or not doping_type or not conc or conc <= 0:
            return Eg

        # Simple model: doping effect increases/decreases Eg slightly
        # based on concentration (logarithmic scaling)
        delta = 0

        # n-type usually reduces band gap slightly
        if doping_type == "n":
            delta = -0.015 * np.log10(max(conc, 1))

        # p-type usually increases band gap slightly
        elif doping_type == "p":
            delta = +0.015 * np.log10(max(conc, 1))

        return float(Eg + delta)

    # -------------------------------------------------------
    #          INVERSE: suggest materials by target Eg
    # -------------------------------------------------------
    def suggest_materials_from_bandgap(self, target, radius_nm, epsilon_r, crystal_structure=None, top_k=3):

        preds = []
        for _, row in self.df.iterrows():
            name = row["material"]
            eg = self.predict_bandgap(name, radius_nm, epsilon_r, crystal_structure)
            score = 1.0 / (1 + abs(eg - target))
            preds.append((name, score))

        preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
        return preds_sorted[:top_k], self.df
