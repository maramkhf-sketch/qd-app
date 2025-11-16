# qd_pipeline.py — Hybrid QD System (CSV + Brus + Optional Doping)

import numpy as np
import pandas as pd

class QDSystem:
    def __init__(self, df):
        """
        df MUST contain:
        material, band_gap_eV, radius_nm, epsilon_r, crystal_structure
        """
        self.df = df

    # -------------------------------------------------------
    # PHYSICAL BRUS EQUATION (BASELINE ESTIMATE)
    # -------------------------------------------------------
    def brus_bandgap(self, Eg_bulk, radius_nm, eps_r):
        """
        Simplified Brus equation
        Used when user picks any material (CSV based)
        """
        r = radius_nm * 1e-9  # nm → m
        h2 = (6.626e-34)**2 / (2 * 9.11e-31)  
        confinement = h2 * (np.pi**2) / (r**2)
        coulomb = 1.8 * (1.6e-19)**2 / (4 * np.pi * 8.85e-12 * eps_r * r)

        Eg = Eg_bulk + confinement / 1.6e-19 - coulomb / 1.6e-19
        return float(Eg)

    # -------------------------------------------------------
    # OPTIONAL DOPING SHIFT
    # -------------------------------------------------------
    def apply_doping(self, Eg, dopant, dop_type, conc):
        """
        Toy physics model:
        - Doping shift is weak (0.01–0.15 eV depending on concentration)
        - Only applied when user enters doping values
        """
        if dopant.strip() == "" or dop_type == "None" or conc == 0:
            return Eg   # no doping

        # Convert cm^-3 → scaled shift
        scale = min(conc / 1e20, 1.0)  # avoid insane values
        shift = 0.10 * scale           # 0 → 0.1 eV

        if dop_type == "n-type":
            Eg = Eg - shift
        elif dop_type == "p-type":
            Eg = Eg + shift

        return float(Eg)

    # -------------------------------------------------------
    # GET BULK Eg FROM CSV
    # -------------------------------------------------------
    def get_bulk_Eg(self, material):
        df_mat = self.df[self.df["material"] == material]
        if len(df_mat) == 0:
            # fallback: mean Eg
            return float(self.df["band_gap_eV"].mean())
        return float(df_mat["band_gap_eV"].mean())

    # -------------------------------------------------------
    # FORWARD PREDICTION
    # -------------------------------------------------------
    def predict_forward(self, material, radius_nm, eps_r, cryst, dopant, dop_type, conc):

        Eg_bulk = self.get_bulk_Eg(material)

        # baseline Brus physics
        Eg = self.brus_bandgap(Eg_bulk, radius_nm, eps_r)

        # doping correction
        Eg = self.apply_doping(Eg, dopant, dop_type, conc)

        return Eg

    # -------------------------------------------------------
    # INVERSE PREDICTION
    # -------------------------------------------------------
    def inverse_predict(self, target_eg):
        """
        Return nearest materials from CSV based on band gap proximity
        """
        df = self.df.copy()
        df["difference"] = abs(df["band_gap_eV"] - target_eg)
        df_sorted = df.sort_values("difference").head(5)

        return df_sorted[["material", "band_gap_eV", "radius_nm", "epsilon_r", "crystal_structure"]]

    # -------------------------------------------------------
    # CURVE PLOT
    # -------------------------------------------------------
    def plot_curve(self, material, eps_r, cryst):
        radii = np.linspace(1, 10, 40)
        Eg_bulk = self.get_bulk_Eg(material)

        Eg_curve = [
            self.brus_bandgap(Eg_bulk, r, eps_r)
            for r in radii
        ]
        return radii, Eg_curve
