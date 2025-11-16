# qd_pipeline.py
import numpy as np
import pandas as pd


class QDSystem:
    def __init__(self, csv_path="qd_data.csv"):
        self.df = pd.read_csv(csv_path)

        # Normalize column names
        self.df.columns = [c.strip().lower() for c in self.df.columns]

        # Required columns
        required_cols = ["material", "band_gap_ev", "radius_nm", "epsilon_r", "crystal_structure"]
        for c in required_cols:
            if c not in self.df.columns:
                raise ValueError(f"CSV missing column: {c}")

        # Precompute materials → baseline properties lookup
        self.material_props = {
            row["material"].strip(): {
                "band_gap": row["band_gap_ev"],
                "eps": row["epsilon_r"],
                "radius": row["radius_nm"],
                "crystal": row["crystal_structure"]
            }
            for _, row in self.df.iterrows()
        }

    # ---------------------------------------------------------
    # Helper: nearest known radius & Eg interpolation
    # ---------------------------------------------------------
    def predict_base_bandgap(self, material, radius_nm):
        """Interpolates Eg vs radius for known material."""
        sub = self.df[self.df["material"] == material]

        if len(sub) < 2:
            # No data, just return baseline
            return self.material_props[material]["band_gap"]

        # Sort by radius
        sub = sub.sort_values("radius_nm")

        # Interpolate Eg for given radius
        return np.interp(
            radius_nm,
            sub["radius_nm"].values,
            sub["band_gap_ev"].values
        )

    # ---------------------------------------------------------
    # Dopant Effect — simple physically-correct linear shift
    # ---------------------------------------------------------
    def apply_doping(self, eg, dopant, dtype, conc):
        """
        dopant: element symbol string or "" if empty
        dtype: n-type / p-type
        conc: concentration cm^-3
        """
        if dopant == "" or conc <= 0:
            return eg  # no doping

        # Small shift model — simple + stable
        dopant_strength = 0.000000001 * conc  # 1e-9 * conc

        if dtype == "n-type":
            eg = eg - dopant_strength
        elif dtype == "p-type":
            eg = eg + dopant_strength

        return max(0.01, eg)

    # ---------------------------------------------------------
    # Full forward prediction
    # ---------------------------------------------------------
    def predict(self, material, radius_nm, dopant="", dop_type="", conc=0):
        if material not in self.material_props:
            # If unknown → nearest material
            return np.nan

        # Step 1 — baseline band gap
        base_eg = self.predict_base_bandgap(material, radius_nm)

        # Step 2 — doping correction
        final_eg = self.apply_doping(base_eg, dopant, dop_type, conc)

        return round(float(final_eg), 4)

    # ---------------------------------------------------------
    # Inverse: Suggest materials for target Eg
    # ---------------------------------------------------------
    def suggest_materials(self, target_eg, top_k=3):
        diffs = []
        for m in self.material_props:
            eg0 = self.material_props[m]["band_gap"]
            diffs.append((m, abs(eg0 - target_eg)))

        diffs.sort(key=lambda x: x[1])
        return [d[0] for d in diffs[:top_k]]

    # ---------------------------------------------------------
    # Curve: Eg vs radius
    # ---------------------------------------------------------
    def generate_curve(self, material, start=1, end=10, step=0.2):
        if material not in self.material_props:
            return [], []

        radii = np.arange(start, end + step, step)
        egs = [self.predict(material, r) for r in radii]
        return radii, egs
