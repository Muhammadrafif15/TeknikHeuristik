"""
src/simulated_annealing.py

Modul SIMULATED ANNEALING (SA) -- "Escape Local Optima"
Komponen ketiga dari Hybrid Optimization Core (Gambar 2 paper).

Berbeda dari GA (populasi) dan PSO (swarm), SA bekerja pada SATU solusi
tunggal yang terus "diguncang" (neighbor generation) dan diuji secara
probabilistik -- meniru proses pendinginan logam (annealing) di metalurgi.

Representasi Solusi (sama seperti GA):
    Array integer sepanjang jumlah graduate, tiap elemen = index job.

Parameter SA sesuai Tabel 3 pada paper:
    Initial Temperature (T0)   : 100.0
    Cooling Rate (alpha)       : 0.95 (geometric: T = T * alpha)
    Minimum Temperature        : 0.01
    Reheating Threshold        : 50 iterasi tanpa perbaikan (stagnasi)

Mekanisme Acceptance (Metropolis criterion, untuk kasus MAKSIMASI):
    delta = fitness_baru - fitness_saat_ini
    - jika delta > 0            -> selalu diterima (solusi lebih baik)
    - jika delta <= 0           -> diterima dengan probabilitas exp(delta / T)
      (semakin tinggi suhu T, semakin besar peluang menerima solusi lebih
       buruk -- inilah yang memungkinkan SA "melompat keluar" dari local optimum)

Cara pakai singkat:
    from src.simulated_annealing import run_sa

    best_solution, best_fitness, log = run_sa(df_grad, df_job, max_iterations=500)
"""

import numpy as np
import pandas as pd

from src.GeneticAlgorithm import precompute_fitness_matrix


# ============================================================
# PARAMETER DEFAULT SA (sesuai Tabel 3 paper)
# ============================================================
SA_PARAMS = {
    "initial_temperature": 100.0,
    "cooling_rate": 0.95,       # geometric cooling: T = T * cooling_rate
    "min_temperature": 0.01,
    "reheat_threshold": 50,     # iterasi stagnan sebelum reheating
    "reheat_temperature": 100.0,  # suhu yang dipakai saat reheating
    "genes_to_perturb": 1,      # jumlah gen yang diubah tiap neighbor generation
}


# ============================================================
# TAHAP 1: EVALUASI SATU SOLUSI
# ============================================================
def evaluate_solution(solution: np.ndarray, fitness_matrix: np.ndarray) -> float:
    """Menghitung fitness (rata-rata F(x)) dari satu solusi assignment."""
    n_graduates = len(solution)
    grad_indices = np.arange(n_graduates)
    return float(fitness_matrix[grad_indices, solution].mean())


# ============================================================
# TAHAP 2: NEIGHBOR GENERATION
# (mengguncang solusi -> mengubah sejumlah kecil gen secara acak)
# ============================================================
def generate_neighbor(solution: np.ndarray, n_jobs: int,
                       genes_to_perturb: int = 1) -> np.ndarray:
    """
    Membuat solusi tetangga (neighbor) dengan mengubah `genes_to_perturb`
    posisi graduate secara acak ke job index baru yang berbeda.
    """
    neighbor = solution.copy()
    n_graduates = len(solution)

    positions_to_change = np.random.choice(
        n_graduates, size=min(genes_to_perturb, n_graduates), replace=False
    )

    for pos in positions_to_change:
        current_job = neighbor[pos]
        # pilih job baru yang berbeda dari job saat ini
        new_job = np.random.randint(0, n_jobs)
        while new_job == current_job and n_jobs > 1:
            new_job = np.random.randint(0, n_jobs)
        neighbor[pos] = new_job

    return neighbor


# ============================================================
# TAHAP 3: ACCEPTANCE CRITERION (Metropolis, untuk MAKSIMASI)
# ============================================================
def acceptance_probability(delta: float, temperature: float) -> float:
    """
    delta = fitness_baru - fitness_saat_ini
    - delta > 0  -> probabilitas 1.0 (selalu diterima, solusi lebih baik)
    - delta <= 0 -> probabilitas exp(delta / T) (bisa diterima meski lebih buruk)
    """
    if delta > 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return float(np.exp(delta / temperature))


# ============================================================
# TAHAP 4: COOLING SCHEDULE (Geometric)
# ============================================================
def cool_down(temperature: float, cooling_rate: float) -> float:
    """T_baru = T_saat_ini * cooling_rate (geometric cooling)."""
    return temperature * cooling_rate


# ============================================================
# FUNGSI UTAMA: MENJALANKAN SA
# ============================================================
def run_sa(
    df_grad: pd.DataFrame,
    df_job: pd.DataFrame,
    initial_solution: np.ndarray = None,
    max_iterations: int = 500,
    params: dict = None,
    fitness_matrix: np.ndarray = None,
    verbose: bool = True,
) -> tuple:
    """
    Menjalankan Simulated Annealing murni untuk mencari solusi terbaik,
    dengan mekanisme reheating jika stagnan.

    Parameters
    ----------
    initial_solution : np.ndarray, optional
        Solusi awal (mis. hasil akhir GA/PSO) -- titik "serah terima" pada
        hybrid optimizer nantinya. Jika None, mulai dari solusi acak.

    Returns
    -------
    best_solution : np.ndarray
        Solusi terbaik yang ditemukan sepanjang proses (assignment job index).
    best_fitness : float
        Nilai fitness dari best_solution.
    convergence_log : list of dict
        Riwayat {iteration, current_fitness, best_fitness, temperature,
        accepted, reheated} tiap iterasi.
    """
    p = {**SA_PARAMS, **(params or {})}
    n_graduates = len(df_grad)
    n_jobs = len(df_job)

    if fitness_matrix is None:
        if verbose:
            print("[SA] Menghitung fitness matrix (sekali saja)...")
        fitness_matrix = precompute_fitness_matrix(df_grad, df_job)

    # --- Solusi awal
    if initial_solution is None:
        current_solution = np.random.randint(0, n_jobs, size=n_graduates)
    else:
        current_solution = initial_solution.copy()

    current_fitness = evaluate_solution(current_solution, fitness_matrix)
    best_solution = current_solution.copy()
    best_fitness = current_fitness

    temperature = p["initial_temperature"]
    stagnation_counter = 0
    convergence_log = []

    for it in range(max_iterations):
        neighbor = generate_neighbor(current_solution, n_jobs, p["genes_to_perturb"])
        neighbor_fitness = evaluate_solution(neighbor, fitness_matrix)

        delta = neighbor_fitness - current_fitness
        prob_accept = acceptance_probability(delta, temperature)
        accepted = np.random.rand() < prob_accept

        if accepted:
            current_solution = neighbor
            current_fitness = neighbor_fitness

        # --- Update solusi terbaik & counter stagnasi
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_solution = current_solution.copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # --- Reheating jika stagnan terlalu lama
        reheated = False
        if stagnation_counter >= p["reheat_threshold"]:
            temperature = p["reheat_temperature"]
            stagnation_counter = 0
            reheated = True
        else:
            temperature = max(cool_down(temperature, p["cooling_rate"]),
                               p["min_temperature"])

        convergence_log.append({
            "iteration": it,
            "current_fitness": round(current_fitness, 4),
            "best_fitness": round(best_fitness, 4),
            "temperature": round(temperature, 4),
            "accepted": accepted,
            "reheated": reheated,
        })

        if verbose and it % max(1, max_iterations // 10) == 0:
            reheat_tag = " [REHEATED]" if reheated else ""
            print(f"[SA] Iter {it:4d} | Current: {current_fitness:.2f} | "
                  f"Best: {best_fitness:.2f} | T: {temperature:.3f}{reheat_tag}")

    if verbose:
        print(f"\n[SA] Selesai. Fitness terbaik akhir: {best_fitness:.2f}")

    return best_solution, best_fitness, convergence_log


# ============================================================
# MODE STANDALONE (opsional, untuk testing modul ini sendiri)
# ============================================================
if __name__ == "__main__":
    from src.Preprocessing import load_and_preprocess
    from src.GeneticAlgorithm import solution_to_dataframe

    df_grad, df_job = load_and_preprocess(
        grad_path="../data/graduates_dummy.csv",
        job_path="../data/job_postings_dummy.csv",
        verbose=False,
    )

    np.random.seed(42)

    fitness_matrix = precompute_fitness_matrix(df_grad, df_job)
    best_solution, best_fitness, log = run_sa(
        df_grad, df_job, max_iterations=500, fitness_matrix=fitness_matrix
    )

    df_result = solution_to_dataframe(best_solution, df_grad, df_job, fitness_matrix)

    out_path = "../output/sa_result.csv"
    log_path = "../output/sa_convergence_log.csv"

    df_result.to_csv(out_path, index=False)
    pd.DataFrame(log).to_csv(log_path, index=False)

    n_reheats = pd.DataFrame(log)["reheated"].sum()
    print(f"\n[TEST] Jumlah reheating terjadi: {n_reheats}")
    print(f"[TEST] Hasil matching SA disimpan: {out_path}")
    print(f"[TEST] Log konvergensi disimpan  : {log_path}")
    print(f"\n[TEST] Rata-rata fitness solusi terbaik: {best_fitness:.2f}")
    print(df_result.head())