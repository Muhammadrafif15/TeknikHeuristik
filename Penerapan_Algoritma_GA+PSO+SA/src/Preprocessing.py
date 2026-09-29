import pandas as pd

# ============================================================
# KONSTANTA GLOBAL
# ============================================================

# Bobot fungsi fitness multi-objektif, sesuai paper:
# Skills 35%, Culture 20%, Salary 15%, Location 10%, Growth 15%, WorkLife 5%
FITNESS_WEIGHTS = {
    "skills_match": 0.35,
    "cultural_fit": 0.20,
    "salary_alignment": 0.15,
    "location_preference": 0.10,
    "career_growth": 0.15,
    "work_life_balance": 0.05,
}

GRAD_CATEGORICAL_COLS = ["academic_discipline", "career_preference"]

GRAD_NUMERIC_COLS = [
    "gpa", "technical_skills_score", "soft_skills_score",
    "work_experience_months", "salary_expected",
    "pref_culture_collaborative", "pref_culture_hierarchical",
    "pref_culture_fast_paced", "pref_culture_flexible", "pref_culture_formal",
    "importance_career_growth", "importance_worklife_balance",
]

JOB_NUMERIC_COLS = [
    "technical_competency_required", "soft_skill_required", "salary_offered",
    "culture_collaborative", "culture_hierarchical", "culture_fast_paced",
    "culture_flexible", "culture_formal", "career_growth_rating",
    "work_life_balance_rating",
]

CULTURE_DIMS = ["collaborative", "hierarchical", "fast_paced", "flexible", "formal"]


# ============================================================
# TAHAP 1: DATA IMPORT
# ============================================================
def import_data(grad_path: str, job_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_grad_raw = pd.read_csv(grad_path)
    df_job_raw = pd.read_csv(job_path)

    n_missing_grad = df_grad_raw.isnull().sum().sum()
    n_missing_job = df_job_raw.isnull().sum().sum()
    if n_missing_grad > 0 or n_missing_job > 0:
        print(f"[WARNING] Missing values ditemukan -> graduate: {n_missing_grad}, "
              f"job: {n_missing_job}")

    return df_grad_raw, df_job_raw


# ============================================================
# TAHAP 2A: ENCODING
# ============================================================
def encode_graduates(df_grad_raw: pd.DataFrame, all_locations: list[str]) -> pd.DataFrame:
    """One-Hot Encoding untuk academic_discipline, career_preference, lokasi
    preferensi; Label Encoding untuk gender (disimpan terpisah, TIDAK dipakai
    sbg fitur matching untuk menghindari bias)."""
    df = pd.get_dummies(df_grad_raw, columns=GRAD_CATEGORICAL_COLS,
                         prefix=GRAD_CATEGORICAL_COLS)

    for loc in all_locations:
        df[f"loc_pref_{loc}"] = (df_grad_raw["location_preference"] == loc).astype(int)

    gender_map = {"Male": 0, "Female": 1}
    df["gender_code"] = df_grad_raw["gender"].map(gender_map)

    return df


def encode_jobs(df_job_raw: pd.DataFrame, all_locations: list[str]) -> pd.DataFrame:
    """One-Hot Encoding untuk lokasi job. Industry & position tetap disimpan
    sbg teks (tidak di-one-hot, cardinality tinggi & sudah terwakili oleh
    skor technical/soft skill requirement)."""
    df = df_job_raw.copy()
    for loc in all_locations:
        df[f"loc_{loc}"] = (df_job_raw["location"] == loc).astype(int)
    return df


# ============================================================
# TAHAP 2B: NORMALIZATION
# ============================================================
def min_max_normalize(series: pd.Series) -> pd.Series:
    """Normalisasi Min-Max ke rentang [0, 1]."""
    min_val, max_val = series.min(), series.max()
    if max_val == min_val:
        return series * 0
    return (series - min_val) / (max_val - min_val)


def normalize_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Menambahkan kolom baru '<col>_norm' hasil Min-Max scaling untuk
    setiap kolom pada `columns`."""
    df = df.copy()
    for col in columns:
        df[f"{col}_norm"] = min_max_normalize(df[col])
    return df


# ============================================================
# TAHAP 2D: FEATURE VECTOR EXTRACTION
# ============================================================
def extract_feature_vectors(df: pd.DataFrame, skill_cols_norm: list[str],
                             culture_cols_norm: list[str]) -> pd.DataFrame:
    """Membentuk kolom `skill_vector` dan `culture_vector` (list of float,
    sudah ternormalisasi) siap dipakai untuk cosine similarity / euclidean
    distance pada tahap fitness evaluation & optimasi."""
    df = df.copy()
    df["skill_vector"] = df[skill_cols_norm].values.tolist()
    df["culture_vector"] = df[culture_cols_norm].values.tolist()
    return df


# ============================================================
# FUNGSI UTAMA (dipanggil dari main.py)
# ============================================================
def load_and_preprocess(
    grad_path: str = "../data/job_postings_dummy.csv",
    job_path: str = "../data/graduates_dummy.csv",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # --- 1. Import
    df_grad_raw, df_job_raw = import_data(grad_path, job_path)
    if verbose:
        print(f"[IMPORT] Graduate: {df_grad_raw.shape[0]} baris | "
              f"Job: {df_job_raw.shape[0]} baris")

    # --- 2A. Encoding
    all_locations = sorted(
        set(df_grad_raw["location_preference"]).union(set(df_job_raw["location"]))
    )
    df_grad = encode_graduates(df_grad_raw, all_locations)
    df_job = encode_jobs(df_job_raw, all_locations)
    if verbose:
        print(f"[ENCODING] Graduate: {df_grad.shape[1]} kolom | "
              f"Job: {df_job.shape[1]} kolom (setelah one-hot)")

    # --- 2B. Normalization
    df_grad = normalize_columns(df_grad, GRAD_NUMERIC_COLS)
    df_job = normalize_columns(df_job, JOB_NUMERIC_COLS)
    if verbose:
        print("[NORMALIZATION] Selesai (Min-Max scaling, rentang 0-1)")

    # --- 2C. Weighting -> lihat FITNESS_WEIGHTS (konstanta modul ini)
    if verbose:
        print(f"[WEIGHTING] Bobot fitness: {FITNESS_WEIGHTS}")

    # --- 2D. Feature Vector Extraction
    grad_skill_cols = ["technical_skills_score_norm", "soft_skills_score_norm"]
    grad_culture_cols = [f"pref_culture_{c}_norm" for c in CULTURE_DIMS]
    df_grad = extract_feature_vectors(df_grad, grad_skill_cols, grad_culture_cols)

    job_skill_cols = ["technical_competency_required_norm", "soft_skill_required_norm"]
    job_culture_cols = [f"culture_{c}_norm" for c in CULTURE_DIMS]
    df_job = extract_feature_vectors(df_job, job_skill_cols, job_culture_cols)

    if verbose:
        print("[FEATURE EXTRACTION] Kolom 'skill_vector' & 'culture_vector' "
              "berhasil dibentuk")
        print(f"\nTotal fitur akhir -> Graduate: {df_grad.shape[1]} kolom | "
              f"Job: {df_job.shape[1]} kolom")

    return df_grad, df_job


# ============================================================
# MODE STANDALONE (opsional, untuk testing modul ini sendiri)
# ============================================================
if __name__ == "__main__":
    df_grad, df_job = load_and_preprocess(
        grad_path="../data/graduates_dummy.csv",  # sesuaikan nama file sebenarnya
        job_path="../data/job_postings_dummy.csv",  # sesuaikan nama file sebenarnya
    )

    df_grad.to_pickle("../output/preprocessed_graduates.pkl")
    df_job.to_pickle("../output/preprocessed_jobs.pkl")
