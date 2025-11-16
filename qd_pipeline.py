import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class QDSystem:
    def __init__(self):
        self.scaler = None
        self.nn = None
        self.df = None

    def fit(self, df):
        """Train the model using the 700-row dataset."""
        self.df = df.copy()

        # Features for ML
        X = df[["radius_nm", "epsilon_r"]].astype(float)

        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        self.nn = NearestNeighbors(n_neighbors=3)
        self.nn.fit(Xs)

    def _nearest_bandgap(self, radius_nm, epsilon_r):
        """Return weighted bandgap based on nearest materials in dataset."""
        Xq = np.array([[radius_nm, epsilon_r]])
        Xq = self.scaler.transform(Xq)

        dist, idx = self.nn.kneighbors(Xq)
        dist = dist[0]
        idx = idx[0]

        weights = 1 / (dist + 1e-6)
        weights = weights / weights.sum()

        bg = (self.df.iloc[idx]["band_gap_eV"].values * weights).sum()
        return float(bg)

    def predict_bandgap(self, material, radius_nm, epsilon_r, crystal_structure,
                        dopant=None, doping_type=None, doping_conc=None):
        """
        Hybrid model:
        - Always uses radius + epsilon_r + ML for base band gap
        - If doping exists → apply physics correction
        """
        bg = self._nearest_bandgap(radius_nm, epsilon_r)

        # --- Physics Correction for doping (OPTIONAL) ---
        if dopant and doping_type and doping_conc and doping_conc > 0:
            # Normalize concentration (cm^-3 → dimensionless)
            C = float(doping_conc) / 1e18

            if doping_type == "n":
                bg -= 0.05 * C
            elif doping_type == "p":
                bg += 0.05 * C

            # Prevent negative Eg
            bg = max(bg, 0.01)

        return float(bg)

    def suggest_materials_from_bandgap(self, target, radius_nm, epsilon_r,
                                       crystal_structure, top_k=3):
        """Inverse Prediction: Suggest closest materials."""
        Xq = np.array([[radius_nm, epsilon_r]])
        Xq = self.scaler.transform(Xq)

        dist, idx = self.nn.kneighbors(Xq)
        idx = idx[0]

        df_hits = self.df.iloc[idx].copy()
        df_hits["score"] = 1 - abs(df_hits["band_gap_eV"] - target) / max(target, 1e-3)

        df_hits = df_hits.sort_values("score", ascending=False).head(top_k)
        top_list = [(row["material"], float(row["score"])) for _, row in df_hits.iterrows()]

        return top_list, df_hits
