import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, top_k_accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
import joblib

# ===== Dopant Table (from CSV) =====
DOPANT_TABLE = None   # يتم تحميلها من ملف CSV

def load_dopant_table(df):
    global DOPANT_TABLE
    cols = ["dopant", "dopant_valence", "dopant_radius_pm", "dopant_affinity_eV"]
    if all(c in df.columns for c in cols):
        DOPANT_TABLE = df[cols].dropna().drop_duplicates().set_index("dopant")
    else:
        DOPANT_TABLE = pd.DataFrame()

# ===== تركيب المادة =====
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element

TARGET_COL = "band_gap_eV"
ELEMENT_FEATURES = [
    ("Z", "Z"),
    ("X", "X"),
    ("row", "row"),
    ("group", "group"),
    ("atomic_mass", "atomic_mass"),
    ("mendeleev_no", "mendeleev_no"),
]

def composition_to_features(formula: str) -> dict:
    comp = Composition(formula)
    el_fracs = comp.fractional_composition.as_dict()
    feats = {}
    for feat_name, attr in ELEMENT_FEATURES:
        vals = []
        for sym, frac in el_fracs.items():
            e = Element(sym)
            val = getattr(e, attr)
            if val is None:
                continue
            vals.append(val * frac)
        if not vals:
            vals = [0.0]
        feats[f"{feat_name}_mean"] = float(np.sum(vals))
    feats["n_elements"] = len(el_fracs)
    feats["entropy_composition"] = -sum(p * np.log(p + 1e-12) for p in el_fracs.values())
    return feats

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "radius_nm" not in df.columns and "diameter_nm" in df.columns:
        df["radius_nm"] = df["diameter_nm"] / 2.0

    needed = {
        "Z_mean","X_mean","row_mean","group_mean",
        "atomic_mass_mean","mendeleev_no_mean",
        "n_elements","entropy_composition"
    }

    if "material" in df.columns and not needed.issubset(df.columns):
        comp_feats = df["material"].apply(composition_to_features).apply(pd.Series)
        df = pd.concat([df, comp_feats], axis=1)

    if "epsilon_r" not in df.columns:
        df["epsilon_r"] = 9.0
    if "radius_nm" not in df.columns:
        df["radius_nm"] = 3.0

    for c in ["Eg_bulk_eV", "me_eff", "mh_eff"]:
        if c not in df.columns:
            df[c] = np.nan

    # doping placeholders
    if "dopant" not in df.columns:
        df["dopant"] = None
    if "doping_type" not in df.columns:
        df["doping_type"] = None
    if "doping_conc_cm3" not in df.columns:
        df["doping_conc_cm3"] = 0.0

    return df

# ===== تأثير الدوبنق =====
def doping_shift(bandgap, dopant, doping_type, conc):
    if conc is None or conc == 0 or dopant is None:
        return bandgap

    if DOPANT_TABLE is None or dopant not in DOPANT_TABLE.index:
        return bandgap

    row = DOPANT_TABLE.loc[dopant]
    valence = row["dopant_valence"]
    affinity = row["dopant_affinity_eV"]

    shift = 0.0

    # n-type → يقلل الـEg شوي
    if doping_type == "n":
        shift -= 0.05 * np.log10(conc / 1e16) * (affinity / 5)

    # p-type → يزيد الـEg شوي
    if doping_type == "p":
        shift += 0.05 * np.log10(conc / 1e16) * (valence / 3)

    return float(bandgap + shift)

# ===== نماذج =====
def train_forward_regressor(df: pd.DataFrame):
    num_cols = [
        "radius_nm", "epsilon_r",
        "Z_mean", "X_mean", "row_mean", "group_mean",
        "atomic_mass_mean", "mendeleev_no_mean",
        "n_elements", "entropy_composition",
    ]
    cat_cols = ["crystal_structure"]

    df_train = df.dropna(subset=[TARGET_COL])
    X = df_train[num_cols + cat_cols]
    y = df_train[TARGET_COL]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    model = Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(
            n_estimators=600, random_state=42, n_jobs=-1))
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    print(f"[Forward] R²={r2_score(y_te, pred):.3f} | MAE={mean_absolute_error(y_te, pred):.3f} eV")
    return model, pre, num_cols, cat_cols

def train_inverse_selector(df: pd.DataFrame, k_neighbors: int = 5):
    df = df.dropna(subset=[TARGET_COL])

    inv_num = ["band_gap_eV", "radius_nm", "epsilon_r"]
    inv_cat = ["crystal_structure"]
    X = df[inv_num + inv_cat]
    y = df["material"]

    pre_inv = ColumnTransformer([
        ("num", StandardScaler(), inv_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), inv_cat),
    ])

    clf = Pipeline([
        ("pre", pre_inv),
        ("rf", RandomForestClassifier(
            n_estimators=500, random_state=42,
            class_weight="balanced_subsample", n_jobs=-1))
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)
    classes = clf.named_steps["rf"].classes_
    top3 = top_k_accuracy_score(y_te, proba, k=3, labels=classes)

    print(f"[Inverse-Classifier] Top-3 Acc = {top3:.3f}")

    X_embed = pre_inv.fit_transform(X)
    nbrs = NearestNeighbors(
        n_neighbors=k_neighbors, metric="euclidean").fit(X_embed)
    return clf, pre_inv, nbrs, classes

# ===== النظام =====
class QDSystem:
    def __init__(self):
        self.forward = None
        self.forward_pre = None
        self.forward_num = None
        self.forward_cat = None

        self.inverse_clf = None
        self.inverse_pre = None
        self.inverse_nbrs = None
        self.inverse_classes = None

        self.train_df = None

    def fit(self, df: pd.DataFrame):
        load_dopant_table(df)
        self.train_df = prepare_dataframe(df)
        self.forward, self.forward_pre, self.forward_num, self.forward_cat = train_forward_regressor(self.train_df)
        self.inverse_clf, self.inverse_pre, self.inverse_nbrs, self.inverse_classes = train_inverse_selector(self.train_df)

    def predict_bandgap(self, material, radius_nm, epsilon_r, crystal_structure, dopant=None, doping_type=None, doping_conc=None):
        row = pd.DataFrame([{
            "material": material,
            "radius_nm": radius_nm,
            "epsilon_r": epsilon_r,
            "crystal_structure": crystal_structure
        }])

        row = prepare_dataframe(row)
        X = row[self.forward_num + self.forward_cat]

        base = float(self.forward.predict(X)[0])
        shifted = doping_shift(base, dopant, doping_type, doping_conc)
        return shifted
