# qd_pipeline.py — Clean version with optional doping support (metadata only)

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class QDSystem:
    """
    Machine-learning system for quantum-dot bandgap prediction.
    - Uses KNN regression.
    - Inputs: material, radius, epsilon_r, crystal_structure.
    - Doping fields are accepted but not used in the ML model (metadata only).
    """

    def __init__(self):
        self.model = None
        self.material_encoder = {}
        self.crystal_encoder = {}

    # -------------------------------
    # Utility encoders
    # -------------------------------
    def encode_material(self, material: str) -> int:
        material = material.strip()
        if material not in self.material_encoder:
            self.material_encoder[material] = len(self.material_encoder)
        return self.material_encoder[material]

    def encode_crystal(self, crys: str) -> int:
        crys = crys.strip()
        if crys not in self.crystal_encoder:
            self.crystal_encoder[crys] = len(self.crystal_encoder)
        return self.crystal_encoder[crys]

    # -------------------------------
    # Fit model
    # -------------------------------
    def fit(self, df: pd.DataFrame):
        # Expecting columns:
        # ['material','radius_nm','epsilon_r','crystal_structure','band_gap_eV']

        df = df.dropna(subset=["material","radius_nm","epsilon_r",
                               "crystal_structure","band_gap_eV"])

        X = []
        y = []

        for _, row in df.iterrows():
            X.append([
                self.encode_material(row["material"]),
                float(row["radius_nm"]),
                float(row["epsilon_r"]),
                self.encode_crystal(row["crystal_structure"])
            ])
            y.append(float(row["band_gap_eV"]))

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        # ML pipeline: scale → KNN
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=5))
        ])

        self.model.fit(X, y)

    # -------------------------------
    # Predict Band Gap
    # -------------------------------
    def predict_bandgap(self, material, radius_nm, epsilon_r, crystal_structure,
                        dopant=None, doping_type=None, doping_conc_cm3=None):
        """
        Doping fields are accepted but *not used* in the ML model.
        They can be integrated later for physics-based corrections.
        """

        X = np.array([[
            self.encode_material(material),
            float(radius_nm),
            float(epsilon_r),
            self.encode_crystal(crystal_structure)
        ]], dtype=float)

        return float(self.model.predict(X)[0])

    # -------------------------------
    # Inverse: Suggest materials for a target bandgap
    # -------------------------------
    def suggest_materials_from_bandgap(self, target_Eg, radius_nm, epsilon_r,
                                       crystal_structure, top_k=3):
        """
        Find the closest materials in dataset space w.r.t predicted bandgap.
        """

        # Generate search set: every known material
        mats = list(self.material_encoder.keys())

        scored = []
        for mat in mats:
            pred = self.predict_bandgap(mat, radius_nm, epsilon_r, crystal_structure)
            score = 1 / (1 + abs(pred - target_Eg))  # similarity
            scored.append((mat, score))

        # Sort descending by score
        scored = sorted(scored, key=lambda x: x[1], reverse=True)
        topk = scored[:top_k]

        # Produce a small nearest-neighbor dataframe for display
        nn_rows = []
        for mat, sc in topk:
            pred = self.predict_bandgap(mat, radius_nm, epsilon_r, crystal_structure)
            nn_rows.append({
                "material": mat,
                "band_gap_eV": pred,
                "radius_nm": radius_nm,
                "epsilon_r": epsilon_r,
                "crystal_structure": crystal_structure
            })

        nn_df = pd.DataFrame(nn_rows)
        return topk, nn_df
