"""
src/fitness_function.py

Modul MULTI-OBJECTIVE CAREER FITNESS FUNCTION
Sesuai kotak "MULTI-OBJECTIVE CAREER FITNESS FUNCTION" pada Gambar 2
(Conceptual Framework) di paper:

    F(x) = w1*f1 + w2*f2 + w3*f3 + w4*f4 + w5*f5 + w6*f6

    f1 Skills Match      (w=0.35) -> cosine similarity(G, J)
    f2 Cultural Fit      (w=0.20) -> 100% - (euclidean_dist / d_max)*100%
    f3 Salary Alignment  (w=0.15) -> piecewise dari R = salary_offered/salary_expected
    f4 Location Pref     (w=0.10) -> kecocokan preferensi lokasi graduate vs job
    f5 Career Growth     (w=0.15) -> skala Likert graduate vs rating job
    f6 Work-Life Balance (w=0.05) -> skala Likert graduate vs rating job

"""

import numpy as np
import pandas as pd

from src.Preprocessing import FITNESS_WEIGHTS, CULTURE_DIMS

# ============================================================
# JARAK EUCLIDEAN MAKSIMUM untuk cultural fit
# (5 dimensi budaya, masing-masing sudah dinormalisasi ke rentang 0-1,
#  sehingga jarak maksimum antar 2 titik adalah sqrt(5) * 1 = akar(5))
# ============================================================
D_MAX_CULTURE = np.sqrt(len(CULTURE_DIMS))


# ============================================================
# f1: SKILLS MATCH -> Cosine Similarity
# ============================================================
def skills_match(grad_skill_vector: list, job_skill_vector: list) -> float:
    """
    f1(x) = (G . J) / (||G|| * ||J||) * 100%
    Menggunakan skill_vector yang sudah ternormalisasi (technical, soft skill).
    """
    vec_a = np.array(grad_skill_vector)
    vec_b = np.array(job_skill_vector)

    norm_product = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if norm_product == 0:
        return 0.0

    cosine_sim = np.dot(vec_a, vec_b) / norm_product
    return float(np.clip(cosine_sim, 0, 1) * 100)


# ============================================================
# f2: CULTURAL FIT -> Euclidean Distance (5 dimensi budaya organisasi)
# ============================================================
def cultural_fit(grad_culture_vector: list, job_culture_vector: list) -> float:
    """
    f2(x) = 100% - (d(Cg, Cj) / d_max) * 100%
    Menggunakan culture_vector (5 dimensi, ternormalisasi 0-1).
    """
    vec_g = np.array(grad_culture_vector)
    vec_j = np.array(job_culture_vector)

    distance = np.linalg.norm(vec_g - vec_j)
    score = 100 - (distance / D_MAX_CULTURE) * 100
    return float(max(score, 0))


# ============================================================
# f3: SALARY ALIGNMENT -> Piecewise Function
# ============================================================
def salary_alignment(salary_offered: float, salary_expected: float) -> float:
    """
    R = Salary Offered / Salary Expected
    f3(x) = 100%      jika R >= 1.0
            R * 100%   jika 0.8 <= R < 1.0
            50% * R    jika R < 0.8
    """
    if salary_expected <= 0:
        return 0.0

    R = salary_offered / salary_expected

    if R >= 1.0:
        return 100.0
    elif R >= 0.8:
        return R * 100
    else:
        return 50 * R


# ============================================================
# f4: LOCATION PREFERENCE
# ============================================================
def location_preference(grad_location_pref: str, job_location: str) -> float:
    """
    Skor kecocokan lokasi:
    - 100%  jika preferensi graduate "Any" ATAU sama persis dengan lokasi job
    - 70%   jika salah satu pihak "Remote" (kompromi jarak jauh)
    - 40%   jika lokasi berbeda & tidak ada yang "Remote"
    """
    if grad_location_pref == "Any" or grad_location_pref == job_location:
        return 100.0
    elif grad_location_pref == "Remote" or job_location == "Remote":
        return 70.0
    else:
        return 40.0


# ============================================================
# f5 & f6: CAREER GROWTH / WORK-LIFE BALANCE -> Likert Scale Matching
# ============================================================
def likert_match(importance: float, rating: float, scale_max: float = 5.0) -> float:
    """
    Basis skor dari rating perusahaan (dikonversi ke persentase),
    dikurangi penalti jika tingkat kepentingan (importance) graduate
    lebih tinggi daripada rating yang ditawarkan perusahaan.
    """
    base_score = (rating / scale_max) * 100
    penalty = max(0.0, (importance - rating)) * 5
    return float(max(base_score - penalty, 0))


# ============================================================
# FUNGSI UTAMA: OVERALL FITNESS F(x)
# ============================================================
def calculate_overall_fitness(
    grad_row: pd.Series,
    job_row: pd.Series,
    weights: dict = None,
) -> tuple[float, dict]:
    """
    Menghitung F(x) = Sigma(wi * fi(x)) untuk SATU pasangan graduate-job.

    Parameters
    ----------
    grad_row : pandas.Series
        Satu baris dari df_grad hasil `load_and_preprocess()`. Wajib punya
        kolom: skill_vector, culture_vector, salary_expected,
        location_preference, importance_career_growth,
        importance_worklife_balance.
    job_row : pandas.Series
        Satu baris dari df_job hasil `load_and_preprocess()`. Wajib punya
        kolom: skill_vector, culture_vector, salary_offered, location,
        career_growth_rating, work_life_balance_rating.
    weights : dict, optional
        Bobot fitness. Default memakai FITNESS_WEIGHTS dari data_preprocessing.

    Returns
    -------
    overall_score : float
        Skor F(x) akhir (0-100), inilah yang dipakai sbg fungsi objektif GA/PSO/SA.
    components : dict
        Rincian skor tiap komponen (f1-f6), berguna untuk logging/analisis.
    """
    if weights is None:
        weights = FITNESS_WEIGHTS

    f1 = skills_match(grad_row["skill_vector"], job_row["skill_vector"])
    f2 = cultural_fit(grad_row["culture_vector"], job_row["culture_vector"])
    f3 = salary_alignment(job_row["salary_offered"], grad_row["salary_expected"])
    f4 = location_preference(grad_row["location_preference"], job_row["location"])
    f5 = likert_match(grad_row["importance_career_growth"], job_row["career_growth_rating"])
    f6 = likert_match(grad_row["importance_worklife_balance"], job_row["work_life_balance_rating"])

    overall_score = (
        weights["skills_match"] * f1
        + weights["cultural_fit"] * f2
        + weights["salary_alignment"] * f3
        + weights["location_preference"] * f4
        + weights["career_growth"] * f5
        + weights["work_life_balance"] * f6
    )

    components = {
        "skills_match_f1": round(f1, 2),
        "cultural_fit_f2": round(f2, 2),
        "salary_alignment_f3": round(f3, 2),
        "location_pref_f4": round(f4, 2),
        "career_growth_f5": round(f5, 2),
        "worklife_balance_f6": round(f6, 2),
    }

    return round(overall_score, 2), components


# ============================================================
# FUNGSI BATCH: HITUNG FITNESS UNTUK BANYAK PASANGAN SEKALIGUS
# (berguna untuk evaluasi populasi GA / swarm PSO / baseline brute-force)
# ============================================================
def calculate_fitness_matrix(
    df_grad: pd.DataFrame,
    df_job: pd.DataFrame,
    weights: dict = None,
) -> pd.DataFrame:
    """
    Menghitung fitness untuk SEMUA kombinasi graduate x job (cross join).
    Mengembalikan DataFrame panjang (long format) dengan kolom fitness lengkap.

    Catatan: cocok untuk dataset kecil-menengah (mis. 200 graduate x 25 job
    = 5000 baris). Untuk skala jauh lebih besar, sebaiknya dipanggil per-batch
    atau hanya untuk populasi kandidat solusi GA/PSO, bukan brute-force penuh.
    """
    if weights is None:
        weights = FITNESS_WEIGHTS

    results = []
    for _, grad_row in df_grad.iterrows():
        for _, job_row in df_job.iterrows():
            score, comp = calculate_overall_fitness(grad_row, job_row, weights)
            results.append({
                "graduate_id": grad_row["graduate_id"],
                "job_id": job_row["job_id"],
                "company_id": job_row["company_id"],
                **comp,
                "overall_fitness_score": score,
            })

    return pd.DataFrame(results)


# ============================================================
# MODE STANDALONE (opsional, untuk testing modul ini sendiri)
# ============================================================
if __name__ == "__main__":
    from src.Preprocessing import load_and_preprocess

    df_grad, df_job = load_and_preprocess(
        grad_path="../data/graduates_dummy.csv",
        job_path="../data/job_postings_dummy.csv",
        verbose=False,
    )

    # --- Tes 1: fitness untuk SATU pasangan saja
    score, components = calculate_overall_fitness(df_grad.iloc[0], df_job.iloc[0])
    print("[TEST] Fitness graduate pertama vs job pertama:")
    print(f"  Overall score : {score}")
    print(f"  Rincian       : {components}")

    # --- Tes 2: fitness matrix untuk semua kombinasi
    df_fitness_matrix = calculate_fitness_matrix(df_grad, df_job)
    print(f"\n[TEST] Total kombinasi dihitung: {len(df_fitness_matrix)}")
    print(df_fitness_matrix.describe()["overall_fitness_score"])
