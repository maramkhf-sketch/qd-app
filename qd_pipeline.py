# qd_pipeline.py

import numpy as np
import pandas as pd
from joblib import load

class QDSystem:
    def __init__(self, csv_path="qd_data.csv"):
        self.df = pd.read_csv(csv_path)

        # مواد النظام الأساسية
        self.materials = sorted(self.df["material"].unique())

        # ماب للكرستال
        self.crystal_map = {"ZB": 0, "WZ": 1}

    # -------------------------------
    # 1) أقرب مادة إذا اليوزر كتب مادة غير موجودة
    # -------------------------------
    def find_closest_material(self, user_material):
        user_material = user_material.strip().lower()
        best = None
        best_score = -1

        for m in self.materials:
            score = sum(a == b for a, b in zip(m.lower(), user_material))
            if score > best_score:
                best_score = score
                best = m
        return best

    # -------------------------------
    # 2) الباند جاب الأساسي من الملف
    # -------------------------------
    def get_base_bandgap(self, material):
        row = self.df[self.df["material"] == material]
        if row.empty:
            material = self.find_closest_material(material)
            row = self.df[self.df["material"] == material]
        return float(row["band_gap_eV"].values[0])

    # -------------------------------
    # 3) تأثير الدوبنق
    # -------------------------------
    def doping_shift(self, dopant, doping_type, concentration):
        if dopant.strip() == "":
            return 0.0

        concentration = max(float(concentration), 0)

        if doping_type == "n-type":
            return 0.04 * np.log10(concentration + 1)
        elif doping_type == "p-type":
            return -0.04 * np.log10(concentration + 1)
        return 0.0

    # -------------------------------
    # 4) معادلة Brus + الدوبنق
    # -------------------------------
    def predict_bandgap(self, material, radius_nm, eps_r, crystal_structure,
                        dopant="", doping_type="", concentration=0):

        material = material if material in self.materials else self.find_closest_material(material)

        base_Eg = self.get_base_bandgap(material)

        r = float(radius_nm) * 1e-9
        e2 = 1.44
        eff_mass = 0.13

        brus = base_Eg + ((np.pi ** 2) * (6.626e-34**2)) / (2 * eff_mass * 9.11e-31 * r**2) \
               - (1.8 * e2)/(eps_r * r * 1e9)

        delta = self.doping_shift(dopant, doping_type, concentration)

        return float(brus + delta)
