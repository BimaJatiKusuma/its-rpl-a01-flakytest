"""
Modul Rekayasa Fitur Project-Agnostic untuk Eksperimen Alternatif 3:
- Transformasi Fitur Rasio Relatif (Project-Agnostic Ratios)
- Transformasi Logaritmik (log(1+x)) untuk Mengurangi Skewness Ukuran Absolut
- Reduksi Dimensi PCA untuk Multikolinearitas Code Churn (hIndex)
- Pipeline zero data leakage: Scaler, PCA, dan Seleksi Fitur di-fit HANYA pada data latih.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# 23 Fitur FlakeFlagger Asli
TEST_SMELLS: List[str] = [
    "assertion-roulette",
    "conditional-test-logic",
    "eager-test",
    "fire-and-forget",
    "indirect-testing",
    "mystery-guest",
    "resource-optimism",
    "test-run-war",
]

EXECUTION_COVERAGE_RAW: List[str] = [
    "testLength",
    "numAsserts",
    "numCoveredLines",
    "ExecutionTime",
    "projectSourceLinesCovered",
    "projectSourceClassesCovered",
    "num_third_party_libs",
]

CHURN_HINDEX_RAW: List[str] = [
    "hIndexModificationsPerCoveredLine_window5",
    "hIndexModificationsPerCoveredLine_window10",
    "hIndexModificationsPerCoveredLine_window25",
    "hIndexModificationsPerCoveredLine_window50",
    "hIndexModificationsPerCoveredLine_window75",
    "hIndexModificationsPerCoveredLine_window100",
    "hIndexModificationsPerCoveredLine_window500",
    "hIndexModificationsPerCoveredLine_window10000",
]

RATIO_NAMES: List[str] = [
    "assert_density",
    "coverage_ratio",
    "class_coverage_ratio",
    "time_per_assert",
    "time_per_line",
]

LOG_FEATURE_NAMES: List[str] = [
    "log_ExecutionTime",
    "log_testLength",
    "log_numCoveredLines",
    "log_projectSourceLinesCovered",
    "log_projectSourceClassesCovered",
    "log_numAsserts",
]


def compute_relative_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung 5 rasio metrik relatif project-agnostic.
    Menggunakan epsilon (+1) pada penyebut untuk stabilitas numerik dan mencegah pembagian dengan nol.
    """
    ratios = pd.DataFrame(index=df.index)
    
    # 1. Kerapatan asersi per baris tes
    ratios["assert_density"] = df["numAsserts"] / (df["testLength"] + 1.0)
    
    # 2. Rasio cakupan baris produksi per baris tes
    ratios["coverage_ratio"] = df["numCoveredLines"] / (df["testLength"] + 1.0)
    
    # 3. Rasio cakupan kelas produksi per pustaka pihak ketiga
    ratios["class_coverage_ratio"] = df["projectSourceClassesCovered"] / (df["num_third_party_libs"] + 1.0)
    
    # 4. Waktu eksekusi per asersi
    ratios["time_per_assert"] = df["ExecutionTime"] / (df["numAsserts"] + 1.0)
    
    # 5. Waktu eksekusi per baris produksi tercover
    ratios["time_per_line"] = df["ExecutionTime"] / (df["numCoveredLines"] + 1.0)
    
    return ratios


def compute_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menerapkan transformasi log(1+x) pada metrik absolut yang right-skewed.
    """
    log_df = pd.DataFrame(index=df.index)
    log_df["log_ExecutionTime"] = np.log1p(np.maximum(0.0, df["ExecutionTime"].values))
    log_df["log_testLength"] = np.log1p(np.maximum(0.0, df["testLength"].values))
    log_df["log_numCoveredLines"] = np.log1p(np.maximum(0.0, df["numCoveredLines"].values))
    log_df["log_projectSourceLinesCovered"] = np.log1p(np.maximum(0.0, df["projectSourceLinesCovered"].values))
    log_df["log_projectSourceClassesCovered"] = np.log1p(np.maximum(0.0, df["projectSourceClassesCovered"].values))
    log_df["log_numAsserts"] = np.log1p(np.maximum(0.0, df["numAsserts"].values))
    return log_df


class PCAChurnTransformer:
    """
    Transformer PCA untuk 8 window code churn (hIndex).
    StandardScaler dan PCA di-fit HANYA pada data latih fold untuk memastikan zero data leakage.
    """
    def __init__(self, variance_threshold: float = 0.95):
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.variance_threshold, svd_solver="full")
        self.is_fitted = False
        self.feature_names_in_: List[str] = []
        self.component_names_: List[str] = []
        self.explained_variance_ratio_: np.ndarray = np.array([])
        self.components_: np.ndarray = np.array([])

    def fit(self, X_hindex: pd.DataFrame) -> "PCAChurnTransformer":
        self.feature_names_in_ = list(X_hindex.columns)
        X_scaled = self.scaler.fit_transform(X_hindex)
        self.pca.fit(X_scaled)
        self.is_fitted = True
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.components_ = self.pca.components_
        n_comp = self.pca.n_components_
        self.component_names_ = [f"pca_churn_{i+1}" for i in range(n_comp)]
        return self

    def transform(self, X_hindex: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("PCAChurnTransformer belum di-fit.")
        X_scaled = self.scaler.transform(X_hindex)
        X_trans = self.pca.transform(X_scaled)
        return pd.DataFrame(X_trans, columns=self.component_names_, index=X_hindex.index)

    def fit_transform(self, X_hindex: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X_hindex).transform(X_hindex)


class FeatureSetPipeline:
    """
    Pipeline lengkap pembangunan 4 Set Fitur:
    - F1_Original: 23 fitur FlakeFlagger asli
    - F2_Ratios_Only: 23 fitur asli + 5 rasio relatif (28 fitur)
    - F3_Project_Agnostic: 5 rasio + 8 test smells + log-transformed size + PCA churn components + num_third_party_libs
    - F4_Feature_Selection: Top-K fitur terseleksi dari F3 menggunakan ANOVA F-score pada data latih fold
    """
    def __init__(self, k_features: int = 12, pca_variance_threshold: float = 0.95):
        self.k_features = k_features
        self.pca_variance_threshold = pca_variance_threshold
        self.pca_transformer: Optional[PCAChurnTransformer] = None
        self.selector: Optional[SelectKBest] = None
        self.selected_f4_feature_names: List[str] = []

    def fit(self, train_df: pd.DataFrame, y_train: pd.Series) -> "FeatureSetPipeline":
        # Fit PCA pada 8 churn windows di training fold
        self.pca_transformer = PCAChurnTransformer(variance_threshold=self.pca_variance_threshold)
        self.pca_transformer.fit(train_df[CHURN_HINDEX_RAW])

        # Bangun representasi F3 training untuk fit Feature Selector F4
        X3_train = self._build_f3(train_df, is_train=True)
        
        # Fit SelectKBest (ANOVA F-value) hanya pada data latih
        k = min(self.k_features, X3_train.shape[1])
        self.selector = SelectKBest(score_func=f_classif, k=k)
        self.selector.fit(X3_train, y_train)
        selected_mask = self.selector.get_support()
        self.selected_f4_feature_names = [col for col, sel in zip(X3_train.columns, selected_mask) if sel]

        return self

    def _build_f3(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """Membangun set F3 Project-Agnostic Reformed."""
        ratios_df = compute_relative_ratios(df)
        log_df = compute_log_transforms(df)
        smells_df = df[TEST_SMELLS].copy()
        third_party_df = df[["num_third_party_libs"]].copy()

        if is_train:
            pca_df = self.pca_transformer.transform(df[CHURN_HINDEX_RAW])
        else:
            pca_df = self.pca_transformer.transform(df[CHURN_HINDEX_RAW])

        f3_df = pd.concat([ratios_df, smells_df, log_df, third_party_df, pca_df], axis=1)
        return f3_df

    def transform(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Menghasilkan representasi fitur untuk set F1, F2, F3, F4."""
        # F1: 23 fitur asli
        f1_df = df[TEST_SMELLS + EXECUTION_COVERAGE_RAW + CHURN_HINDEX_RAW].copy()

        # F2: F1 + 5 rasio relatif
        ratios_df = compute_relative_ratios(df)
        f2_df = pd.concat([f1_df, ratios_df], axis=1)

        # F3: Project-Agnostic Reformed
        f3_df = self._build_f3(df, is_train=False)

        # F4: Sub-fitur terseleksi dari F3
        f4_df = f3_df[self.selected_f4_feature_names].copy()

        return {
            "F1_Original": f1_df,
            "F2_Ratios_Only": f2_df,
            "F3_Project_Agnostic": f3_df,
            "F4_Feature_Selection": f4_df,
        }


def compute_vif_dataframe(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung Variance Inflation Factor (VIF) untuk setiap fitur dalam DataFrame.
    Menambahkan konstanta (intercept) untuk evaluasi multikolinearitas yang valid.
    """
    # Bersihkan NaN / Inf
    df_clean = df_features.replace([np.inf, -np.inf], np.nan).dropna().copy()
    
    # Hapus kolom dengan variansi nol (konstanta)
    numeric_cols = [c for c in df_clean.columns if df_clean[c].std() > 1e-8]
    df_clean = df_clean[numeric_cols]

    if df_clean.shape[1] < 2:
        return pd.DataFrame({"Feature": numeric_cols, "VIF": [1.0] * len(numeric_cols)})

    try:
        X_const = add_constant(df_clean)
        vifs = []
        for i, col in enumerate(X_const.columns):
            if col == "const":
                continue
            try:
                v = variance_inflation_factor(X_const.values, i)
            except Exception:
                v = np.nan
            vifs.append({"Feature": col, "VIF": v})
        return pd.DataFrame(vifs).sort_values(by="VIF", ascending=False).reset_index(drop=True)
    except Exception as e:
        # Fallback jika terjadi singularitas matriks
        vifs = []
        for col in df_clean.columns:
            vifs.append({"Feature": col, "VIF": np.nan})
        return pd.DataFrame(vifs)


if __name__ == "__main__":
    from data_loader import load_dataset
    df, features, projects = load_dataset()
    
    print("[Test FeatureEngineer]")
    pipeline = FeatureSetPipeline(k_features=12, pca_variance_threshold=0.95)
    pipeline.fit(df, df["flaky"])
    transformed = pipeline.transform(df)
    
    for k, v in transformed.items():
        print(f"- {k}: {v.shape[1]} fitur ({v.columns.tolist()[:4]} ...)")
    
    print("\nPCA Components:", pipeline.pca_transformer.component_names_)
    print("Explained Variance Ratio:", np.round(pipeline.pca_transformer.explained_variance_ratio_, 4))
    print("Cumulative Variance:", round(float(np.sum(pipeline.pca_transformer.explained_variance_ratio_)), 4))
    print("Selected F4 Features:", pipeline.selected_f4_feature_names)
