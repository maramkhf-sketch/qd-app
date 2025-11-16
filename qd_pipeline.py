# qd_pipeline.py — Hybrid Physically-Aware + ML System
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib
from pathlib import Path


# ===========================
#   FEATURE EXTRACTION
# ===========================
def extract_features(df):
    df = df.copy()

    # radius
    if "radius_nm" not in df.columns and "diameter_nm" in df.columns:
        df["radius_nm"] = df["diameter_nm"] / 2

    # epsilon_r
    if "epsilon_r" not in df.columns:
        df["epsilon_r"] = 9.0

    # crystal structure
    if "crystal_structure" not in df.columns:
        df["crystal_structure"] = "ZB"

    # fallbacks
    needed = ["Eg_bulk_eV", "me_eff", "mh_eff"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    return df


# ===========================
#    PHYSICAL BRUS MODEL
# ===========================
def brus_Eg(R_nm, Eg_bulk, me_eff, mh_eff, eps_r):
    hbar = 1.054571817e-34
    e = 1.602176634e-19
    eps0 = 8.8541878128e-12
    m0 = 9.10938356e-31

    R = R_nm * 1e-9

    term_conf = (hbar**2 * np.pi**2)/(2*R**2) * (1/(me_eff*m0) + 1/(mh_eff*m0))
    term_coul = (1.8 * e**2)/(4*np.pi*eps0*eps_r*R)

    Eg = Eg_bulk + (term_conf - term_coul)/e
    return Eg


# ===========================
#      QD SYSTEM CLASS
# ===========================
class QDSystem:
    def __init__(self):
        self.forward = None
        self.inv = None
        self.train_df = None
        self.forward_pre = None
        self.inverse_pre = None
        self.inv_classes = None
        self.inv_nbrs = None

    # -----------------------
    def fit(self, df):
        df = extract_features(df)
        self.train_df = df.copy()

        # -------------------------------
        # FORWARD MODEL
        # -------------------------------
        fcols_num = ["radius_nm", "epsilon_r"]
        fcols_cat = ["material", "crystal_structure"]

        df_f = df.dropna(subset=["band_gap_eV"])
        X = df_f[fcols_num + fcols_cat]
        y = df_f["band_gap_eV"]

        pre = ColumnTransformer([
            ("num", StandardScaler(), fcols_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), fcols_cat),
        ])

        model = Pipeline([
            ("pre", pre),
            ("rf", RandomForestRegressor(n_estimators=500, random_state=42))
        ])

        model.fit(X, y)
        self.forward = model
        self.forward_pre = pre

        # -------------------------------
        # INVERSE CLASSIFIER
        # -------------------------------
        icols_num = ["band_gap_eV", "radius_nm", "epsilon_r"]
        icols_cat = ["crystal_structure"]

        df_i = df.dropna(subset=["band_gap_eV"])
        Xi = df_i[icols_num + icols_cat]
        yi = df_i["material"]

        pre_i = ColumnTransformer([
            ("num", StandardScaler(), icols_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), icols_cat),
        ])

        clf = Pipeline([
            ("pre", pre_i),
            ("rf", RandomForestClassifier(n_estimators=400, random_state=42))
        ])

        clf.fit(Xi, yi)
        self.inv = clf
        self.inverse_pre = pre_i
        self.inv_classes = clf.named_steps["rf"].classes_

        # KNN (inverse neighbors)
        Xi_emb = pre_i.transform(Xi)
        nbrs = NearestNeighbors(n_neighbors=3).fit(Xi_emb)
        self.inv_nbrs = nbrs

        return self

    # ---------------------------
    def predict_forward(self, material, R, eps, structure):
        row = pd.DataFrame([{
            "material": material,
            "radius_nm": R,
            "epsilon_r": eps,
            "crystal_structure": structure
        }])
        return float(self.forward.predict(row)[0])

    # ---------------------------
    def predict_inverse(self, Eg, R, eps, structure):
        row = pd.DataFrame([{
            "band_gap_eV": Eg,
            "radius_nm": R,
            "epsilon_r": eps,
            "crystal_structure": structure
        }])

        probs = self.inv.predict_proba(row)[0]
        order = np.argsort(probs)[::-1][:3]
        top = [(self.inv_classes[i], float(probs[i])) for i in order]

        # neighbors
        Xi = self.train_df[["band_gap_eV","radius_nm","epsilon_r","crystal_structure"]]
        Xi_emb = self.inverse_pre.transform(Xi)
        row_emb = self.inverse_pre.transform(row)
        dist, idx = self.inv_nbrs.kneighbors(row_emb, n_neighbors=3)
        nn = self.train_df.iloc[idx[0]]

        return top, nn

    # ---------------------------
    # HYBRID CURVE: ML + Brus + NN (B2)
    # ---------------------------
    def hybrid_curve(self, material, eps, structure,
                     Rmin=1.0, Rmax=10.0, steps=40):

        R_grid = np.linspace(Rmin, Rmax, steps)

        # find bulk / masses for Brus:
        df_m = self.train_df[self.train_df["material"] == material]
        if len(df_m) > 0:
            Eg_bulk = df_m["Eg_bulk_eV"].dropna().mean()
            me = df_m["me_eff"].dropna().mean()
            mh = df_m["mh_eff"].dropna().mean()
            if pd.isna(Eg_bulk): Eg_bulk = 1.8
            if pd.isna(me): me = 0.13
            if pd.isna(mh): mh = 0.45
        else:
            Eg_bulk, me, mh = 1.8, 0.13, 0.45

        Eg_phys = brus_Eg(R_grid, Eg_bulk, me, mh, eps)

        Eg_ml = np.array([
            self.predict_forward(material, float(r), eps, structure)
            for r in R_grid
        ])

        # smooth NN correction
        df_same = df_m.dropna(subset=["band_gap_eV"])
        if len(df_same) > 2:
            R_known = df_same["radius_nm"].values
            Eg_known = df_same["band_gap_eV"].values
            corr = []
            for r in R_grid:
                w = np.exp(-((r - R_known)**2)/(2*0.4**2))
                w = w / (w.sum() + 1e-12)
                corr.append(np.sum(w * Eg_known))
            corr = np.array(corr)
            blend = 0.25*Eg_phys + 0.5*Eg_ml + 0.25*corr
        else:
            blend = 0.4*Eg_phys + 0.6*Eg_ml

        return R_grid, Eg_ml, Eg_phys, blend
