# qd_pipeline.py  — clean, no pymatgen, robust, with optional doping

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class QDSystem:
    """
    Hybrid model:
    - Trains ML regressor on qd_data.csv  (material, radius_nm, epsilon_r, crystal_structure, band_gap_eV)
    - For unknown cases / ML failure, falls back to simple Brus-style trend.
    - Optional doping term (n/p-type) adds a small, physically-reasonable shift.
    """

    def __init__(self):
        self.model = None
        self.df = None
        self._cat_cols = ["material", "crystal_structure"]
        self._num_cols = ["radius_nm", "epsilon_r"]

    # ------------------ Training ------------------ #
    def fit(self, df: pd.DataFrame):
        # Keep only the columns we need
        needed = self._cat_cols + self._num_cols + ["band_gap_eV"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"qd_data.csv is missing columns: {missing}")

        df = df[needed].copy()

        # Clean types
        df["radius_nm"] = pd.to_numeric(df["radius_nm"], errors="coerce")
        df["epsilon_r"] = pd.to_numeric(df["epsilon_r"], errors="coerce")
        df["band_gap_eV"] = pd.to_numeric(df["band_gap_eV"], errors="coerce")

        # Drop rows with missing essentials
        df = df.dropna(subset=["radius_nm", "epsilon_r", "band_gap_eV"])
        if df.empty:
            raise ValueError("No valid rows in qd_data.csv after cleaning.")

        self.df = df.reset_index(drop=True)

        X = self.df[self._cat_cols + self._num_cols]
        y = self.df["band_gap_eV"].astype(float)

        pre = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self._cat_cols),
                ("num", "passthrough", self._num_cols),
            ]
        )

        self.model = Pipeline(
            steps=[
                ("pre", pre),
                ("rf", RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    min_samples_leaf=2,
                    n_jobs=-1,
                )),
            ]
        )

        self.model.fit(X, y)

    # ------------------ Internal helpers ------------------ #
    def _fallback_brus(self, material: str, radius_nm: float,
                       epsilon_r: float, crystal_structure: str) -> float:
        """
        If ML prediction fails / NaN, use a simple Brus-like scaling:
        Eg(R) = Eg_bulk + k / R^2
        with Eg_bulk taken from dataset statistics.
        """
        if self.df is None or self.df.empty:
            # absolute last resort
            return 2.0

        R = max(float(radius_nm), 0.5)

        # Try to use rows for same material; if none, same crystal; else all
        df = self.df
        sub = df[df["material"].str.lower() == str(material).lower()]
        if sub.empty:
            sub = df[df["crystal_structure"].str.lower() == str(crystal_structure).lower()]
        if sub.empty:
            sub = df

        Eg_bulk = float(sub["band_gap_eV"].median())

        # Rough k coefficient based on spread of dataset
        # (just to capture the trend that smaller R => larger Eg)
        R_col = np.maximum(sub["radius_nm"].values.astype(float), 0.5)
        Eg_col = sub["band_gap_eV"].values.astype(float)
        invR2 = 1.0 / (R_col ** 2)
        # Linear fit Eg ~ a + b*(1/R^2)
        A = np.vstack([np.ones_like(invR2), invR2]).T
        try:
            coeffs, *_ = np.linalg.lstsq(A, Eg_col, rcond=None)
            a, b = coeffs
        except Exception:
            a, b = Eg_bulk, 0.5

        Eg_est = float(a + b * (1.0 / (R ** 2)))
        if not np.isfinite(Eg_est):
            Eg_est = Eg_bulk

        return max(Eg_est, 0.0)

    def _apply_doping_shift(self, Eg_base: float,
                            dopant: str | None,
                            doping_type: str | None,
                            doping_conc_cm3: float | None) -> float:
        """
        Very simple Burstein-Moss / band tail approximation:
        ΔEg = s * α * log10(N / N0),  clamped to ±0.3 eV
        where s = +1 (n-type), -1 (p-type)
        """
        if doping_conc_cm3 is None:
            return Eg_base
        try:
            N = float(doping_conc_cm3)
        except Exception:
            return Eg_base

        if N <= 0:
            return Eg_base

        t = (doping_type or "").lower().strip()
        if t not in ("n", "n-type", "p", "p-type"):
            return Eg_base

        sign = 1.0 if t.startswith("n") else -1.0
        N0 = 1e17  # reference density
        alpha = 0.05  # eV per decade

        shift = sign * alpha * np.log10(N / N0)
        shift = float(np.clip(shift, -0.3, 0.3))

        Eg = Eg_base + shift
        return max(Eg, 0.0)

    # ------------------ Public API ------------------ #
    def predict_bandgap(
        self,
        material: str,
        radius_nm: float,
        epsilon_r: float,
        crystal_structure: str,
        dopant: str | None = None,
        doping_type: str | None = None,
        doping_conc_cm3: float | None = None,
    ) -> float:
        if self.model is None or self.df is None:
            raise RuntimeError("QDSystem is not fitted. Call fit(df) first.")

        # Build feature row
        data = {
            "material": [str(material)],
            "radius_nm": [float(radius_nm)],
            "epsilon_r": [float(epsilon_r)],
            "crystal_structure": [str(crystal_structure)],
        }

        try:
            pred = self.model.predict(pd.DataFrame(data))[0]
            Eg_base = float(pred)
        except Exception:
            Eg_base = np.nan

        if not np.isfinite(Eg_base):
            Eg_base = self._fallback_brus(material, radius_nm, epsilon_r, crystal_structure)

        # Doping (optional)
        Eg_total = self._apply_doping_shift(Eg_base, dopant, doping_type, doping_conc_cm3)
        return float(Eg_total)

    def suggest_materials_from_bandgap(
        self,
        target_Eg: float,
        radius_nm: float,
        epsilon_r: float,
        crystal_structure: str,
        top_k: int = 3,
    ):
        """
        Returns:
          topk: list[(material_name, score)]
          nn_df: DataFrame of best matches (for display in Streamlit)
        """
        if self.df is None or self.df.empty:
            return [], pd.DataFrame()

        scores = []
        preds = []

        for _, row in self.df.iterrows():
            Eg_i = self.predict_bandgap(
                material=row["material"],
                radius_nm=radius_nm,
                epsilon_r=epsilon_r,
                crystal_structure=crystal_structure,
                dopant=None,
                doping_type=None,
                doping_conc_cm3=None,
            )
            preds.append(Eg_i)
            # Gaussian score around target
            scores.append(np.exp(-((Eg_i - target_Eg) ** 2) / (2 * 0.25 ** 2)))

        df = self.df.copy()
        df["Eg_pred"] = preds
        df["score"] = scores

        df_sorted = df.sort_values("score", ascending=False)
        top = df_sorted.head(top_k)

        top_list = [(row["material"], float(row["score"])) for _, row in top.iterrows()]

        return top_list, df_sorted.reset_index(drop=True)
