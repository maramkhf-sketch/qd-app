# qd_pipeline.py — Hybrid Quantum-Dot + Doping Model

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element


# ============================================================
#               1)  Feature Engineering from Formula
# ============================================================

ELEMENT_FEATURES = [
    ("Z", "Z"),
    ("X", "X"),
    ("row", "row"),
    ("group", "group"),
    ("atomic_mass", "atomic_mass"),
    ("mendeleev_no", "mendeleev_no"),
]

def composition_to_features(formula: str) -> dict:
    """Extract weighted elemental features using pymatgen."""
    comp = Composition(formula)
    el_fracs = comp.fractional_composition.as_dict()

    feats = {}

    for feat_name, attr in ELEMENT_FEATURES:
        vals = []
        for sym, frac in el_fracs.items():
            e = Element(sym)
            val = getattr(e, attr)
            if val is not None:
                vals.append(val * frac)
        feats[f"{feat_name}_mean"] = float(np.sum(vals)) if vals else 0.0

    feats["n_elements"] = len(el_fracs)
    feats["entropy_composition"] = -sum(p * np.log(p + 1e-12) for p in el_fracs.values())

    return feats


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "radius_nm" not in df.columns and "diameter_nm" in df.columns:
        df["radius_nm"] = df["diameter_nm"] / 2

    if "material" in df.columns:
        needed = {
            "Z_mean","X_mean","row_mean","group_mean",
            "atomic_mass_mean","mendeleev_no_mean",
            "n_elements","entropy_composition"
        }
        if not needed.issubset(df.columns):
            comp_feats = df["material"].apply(composition_to_features).apply(pd.DataFrame)
            df = pd.concat([df, comp_feats], axis=1)

    if "epsilon_r" not in df.columns:
        df["epsilon_r"] = 9.0
    if "radius_nm" not in df.columns:
        df["radius_nm"] = 3.0
    if "crystal_structure" not in df.columns:
        df["crystal_structure"] = "ZB"

    return df


# ============================================================
#                   2)   Base ML Regressor
# ============================================================

TARGET_COL = "band_gap_eV"

NUM_COLS = [
    "radius_nm", "epsilon_r",
    "Z_mean","X_mean","row_mean","group_mean",
    "atomic_mass_mean","mendeleev_no_mean",
    "n_elements","entropy_composition"
]

CAT_COLS = ["crystal_structure"]

def train_forward_regressor(df: pd.DataFrame):
    df = df.dropna(subset=[TARGET_COL])
    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET_COL]

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)
    ])

    model = Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(n_estimators=600, random_state=42))
    ])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(Xtr, ytr)

    return model


# ============================================================
#            3)  Physics Hybrid Model (Quantum Dot)
# ============================================================

# Base Brus equation
def brus_Eg(R_nm, Eg_bulk, me, mh, epsilon_r):
    h = 6.62607015e-34
    hbar = h / (2*np.pi)
    e = 1.602176634e-19
    eps0 = 8.8541878128e-12
    m0 = 9.10938356e-31

    R = R_nm * 1e-9
    term_conf = (hbar**2 * (np.pi**2) / (2*R**2)) * (1/(me*m0) + 1/(mh*m0))
    term_coul = (1.8 * e**2) / (4*np.pi*eps0*epsilon_r*R)

    return Eg_bulk + (term_conf - term_coul) / e


# Doping physical correction
def doping_correction(Eg, doping_type, conc, dopant):
    if doping_type is None or conc is None or conc <= 0:
        return Eg  # no change

    conc = float(conc)

    if doping_type == "n":
        # Burstein-Moss widening
        delta = 0.18 * (np.log10(conc) - 17)
        delta = max(0, min(delta, 0.35))
        return Eg + delta

    if doping_type == "p":
        # band tailing (p-type reduces Eg)
        delta = 0.12 * (np.log10(conc) - 17)
        delta = max(0, min(delta, 0.25))
        return Eg - delta

    return Eg


# ============================================================
#                4) Master system class
# ============================================================

class QDSystem:
    def __init__(self):
        self.model = None
        self.train_df = None

    def fit(self, df: pd.DataFrame):
        df = prepare_dataframe(df)
        self.train_df = df
        self.model = train_forward_regressor(df)

    # ----------------------------------------------------------
    #                  Forward Prediction
    # ----------------------------------------------------------
    def predict_bandgap(self, material, radius_nm, epsilon_r,
                        crystal_structure,
                        dopant=None, doping_type=None, doping_conc_cm3=None):

        df = pd.DataFrame([{
            "material": material,
            "radius_nm": radius_nm,
            "epsilon_r": epsilon_r,
            "crystal_structure": crystal_structure
        }])

        df = prepare_dataframe(df)

        Eg_ml = float(self.model.predict(df[NUM_COLS + CAT_COLS])[0])

        # Physics refinement (Brus)
        Eg_bulk = 1.7 if Eg_ml < 1 else Eg_ml * 0.6
        me, mh = 0.13, 0.45

        Eg_physics = brus_Eg(radius_nm, Eg_bulk, me, mh, epsilon_r)

        # hybrid combine
        Eg_hybrid = 0.55 * Eg_ml + 0.45 * Eg_physics

        # optional doping correction
        Eg_final = doping_correction(Eg_hybrid, doping_type, doping_conc_cm3, dopant)

        return float(Eg_final)


    # ----------------------------------------------------------
    #                    Inverse Suggestions
    # ----------------------------------------------------------
    def suggest_materials_from_bandgap(self, target_Eg, radius_nm, epsilon_r,
                                       crystal_structure, top_k=3):

        df = self.train_df.copy()
        df["pred"] = df.apply(
            lambda row:
                self.predict_bandgap(
                    row["material"],
                    radius_nm,
                    epsilon_r,
                    crystal_structure
                ),
            axis=1
        )

        df["score"] = -abs(df["pred"] - target_Eg)
        df_sorted = df.sort_values("score", ascending=False).head(top_k)

        top_list = [(row["material"], 1/(1+abs(row["pred"]-target_Eg)))
                    for _, row in df_sorted.iterrows()]

        return top_list, df_sorted
