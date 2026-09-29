"""
src/particle_swarm.py

Modul PARTICLE SWARM OPTIMIZATION (PSO) -- "Local Exploitation"
Komponen kedua dari Hybrid Optimization Core (Gambar 2 paper).

Representasi Solusi (Position Encoding):
    PSO klasik bekerja di ruang KONTINU, sedangkan masalah matching di sini
    bersifat DISKRIT (assignment graduate -> index job). Maka setiap partikel
    direpresentasikan sbg vektor posisi kontinu sepanjang jumlah graduate,
    dengan nilai di rentang [0, n_jobs). Untuk dievaluasi fitness-nya, posisi
    ini di-DISKRITKAN (dibulatkan ke bawah / floor) menjadi index job.

    Contoh (3 graduate, 5 job tersedia):
        posisi kontinu = [2.7, 0.3, 4.1]
        --(decode)-->  index job = [2, 0, 4]

Parameter PSO sesuai Tabel 3 pada paper:
    Swarm Size          : 50 partikel
    Inertia Weight (w)  : 0.9 -> 0.4 (linear decay)
    Cognitive Coeff (c1): 2.0
    Social Coeff (c2)   : 2.0
    Velocity Clamp      : +/- 20% dari search space (n_jobs)

Cara pakai singkat:
    from src.particle_swarm import run_pso

    best_position, best_fitness, log = run_pso(df_grad, df_job, iterations=100)
"""

import numpy as np
import pandas as pd

from src.GeneticAlgorithm import precompute_fitness_matrix


# ============================================================
# PARAMETER DEFAULT PSO (sesuai Tabel 3 paper)
# ============================================================
PSO_PARAMS = {
    "swarm_size": 50,
    "inertia_start": 0.9,
    "inertia_end": 0.4,
    "c1": 2.0,   # cognitive coefficient
    "c2": 2.0,   # social coefficient
    "velocity_clamp_ratio": 0.20,  # +/- 20% dari search space
}


# ============================================================
# ENCODING/DECODING POSISI PARTIKEL
# ============================================================
def decode_position(position: np.ndarray, n_jobs: int) -> np.ndarray:
    """
    Mengubah posisi kontinu partikel menjadi index job diskrit (integer).
    Nilai di-clip dulu ke rentang valid [0, n_jobs - 1] sebelum di-floor.
    """
    clipped = np.clip(position, 0, n_jobs - 1e-6)
    return np.floor(clipped).astype(int)


# ============================================================
# TAHAP 1: INISIALISASI SWARM
# ============================================================
def initialize_swarm(swarm_size: int, n_graduates: int, n_jobs: int,
                      seed_positions: np.ndarray = None) -> tuple:
    """
    Membuat posisi & kecepatan awal seluruh partikel.

    Parameters
    ----------
    seed_positions : np.ndarray, optional
        Jika diisi (mis. hasil populasi akhir GA), sebagian swarm akan
        diinisialisasi dari sini -- inilah titik "serah terima" GA -> PSO
        pada hybrid optimizer nantinya.
    """
    positions = np.random.uniform(0, n_jobs, size=(swarm_size, n_graduates))

    if seed_positions is not None:
        n_seed = min(len(seed_positions), swarm_size)
        positions[:n_seed] = seed_positions[:n_seed].astype(float)

    # Kecepatan awal kecil & acak (arah maju-mundur)
    max_velocity = PSO_PARAMS["velocity_clamp_ratio"] * n_jobs
    velocities = np.random.uniform(-max_velocity, max_velocity,
                                    size=(swarm_size, n_graduates))

    return positions, velocities


# ============================================================
# TAHAP 2: EVALUASI FITNESS SWARM
# ============================================================
def evaluate_swarm(positions: np.ndarray, fitness_matrix: np.ndarray,
                    n_jobs: int) -> np.ndarray:
    """
    Menghitung fitness tiap partikel = rata-rata skor F(x) seluruh graduate
    terhadap job hasil decode posisi partikel tsb.
    """
    n_graduates = positions.shape[1]
    grad_indices = np.arange(n_graduates)

    fitness_values = np.array([
        fitness_matrix[grad_indices, decode_position(pos, n_jobs)].mean()
        for pos in positions
    ])
    return fitness_values


# ============================================================
# TAHAP 3: UPDATE KECEPATAN & POSISI (inti PSO)
# ============================================================
def update_velocity(velocities: np.ndarray, positions: np.ndarray,
                     pbest_positions: np.ndarray, gbest_position: np.ndarray,
                     inertia: float, c1: float, c2: float,
                     max_velocity: float) -> np.ndarray:
    """
    v_new = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
    Kecepatan di-clamp ke +/- max_velocity (20% dari search space).
    """
    swarm_size, n_graduates = positions.shape
    r1 = np.random.rand(swarm_size, n_graduates)
    r2 = np.random.rand(swarm_size, n_graduates)

    cognitive = c1 * r1 * (pbest_positions - positions)
    social = c2 * r2 * (gbest_position - positions)  # broadcast gbest ke semua partikel

    new_velocities = inertia * velocities + cognitive + social
    new_velocities = np.clip(new_velocities, -max_velocity, max_velocity)
    return new_velocities


def update_position(positions: np.ndarray, velocities: np.ndarray,
                     n_jobs: int) -> np.ndarray:
    """
    x_new = x + v, lalu di-clip agar tetap berada dalam batas ruang
    pencarian [0, n_jobs).
    """
    new_positions = positions + velocities
    return np.clip(new_positions, 0, n_jobs - 1e-6)


def linear_inertia_decay(iteration: int, max_iterations: int,
                          w_start: float, w_end: float) -> float:
    """Inertia weight menurun linear dari w_start ke w_end sepanjang iterasi."""
    progress = iteration / max(1, max_iterations - 1)
    return w_start - (w_start - w_end) * progress


# ============================================================
# UTILITAS: DIVERSITY SWARM
# (dipakai nanti oleh mekanisme adaptive switching GA<->PSO<->SA)
# ============================================================
def swarm_diversity(positions: np.ndarray, n_jobs: int) -> float:
    """
    Mengukur keberagaman swarm sbg rata-rata standar deviasi posisi tiap
    graduate, dinormalisasi terhadap lebar search space (n_jobs).
    0 = seluruh partikel di posisi identik ; mendekati 1 = sangat beragam.
    """
    std_per_graduate = positions.std(axis=0)
    normalized = std_per_graduate / n_jobs
    return float(np.mean(normalized))


# ============================================================
# FUNGSI UTAMA: MENJALANKAN PSO
# ============================================================
def run_pso(
    df_grad: pd.DataFrame,
    df_job: pd.DataFrame,
    iterations: int = 100,
    params: dict = None,
    fitness_matrix: np.ndarray = None,
    seed_positions: np.ndarray = None,
    verbose: bool = True,
) -> tuple:
    """
    Menjalankan Particle Swarm Optimization murni untuk menyempurnakan
    kombinasi assignment graduate->job.

    Parameters
    ----------
    seed_positions : np.ndarray, optional
        Populasi akhir dari GA (opsional) -- dipakai sbg titik awal sebagian
        swarm, mensimulasikan alur "GA -> PSO" pada hybrid optimizer.

    Returns
    -------
    best_position : np.ndarray
        Posisi (index job) terbaik yang ditemukan (sudah didekode ke integer).
    best_fitness : float
        Nilai fitness dari best_position.
    convergence_log : list of dict
        Riwayat {iteration, best_fitness, avg_fitness, diversity} tiap iterasi.
    """
    p = {**PSO_PARAMS, **(params or {})}
    n_graduates = len(df_grad)
    n_jobs = len(df_job)
    max_velocity = p["velocity_clamp_ratio"] * n_jobs

    if fitness_matrix is None:
        if verbose:
            print("[PSO] Menghitung fitness matrix (sekali saja)...")
        fitness_matrix = precompute_fitness_matrix(df_grad, df_job)

    positions, velocities = initialize_swarm(
        p["swarm_size"], n_graduates, n_jobs, seed_positions
    )

    # Inisialisasi personal best (pbest) & global best (gbest)
    fitness_values = evaluate_swarm(positions, fitness_matrix, n_jobs)
    pbest_positions = positions.copy()
    pbest_fitness = fitness_values.copy()

    gbest_idx = np.argmax(pbest_fitness)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]

    convergence_log = []

    for it in range(iterations):
        inertia = linear_inertia_decay(it, iterations, p["inertia_start"], p["inertia_end"])

        velocities = update_velocity(
            velocities, positions, pbest_positions, gbest_position,
            inertia, p["c1"], p["c2"], max_velocity
        )
        positions = update_position(positions, velocities, n_jobs)

        fitness_values = evaluate_swarm(positions, fitness_matrix, n_jobs)

        # --- Update personal best
        improved = fitness_values > pbest_fitness
        pbest_positions[improved] = positions[improved]
        pbest_fitness[improved] = fitness_values[improved]

        # --- Update global best
        current_best_idx = np.argmax(pbest_fitness)
        if pbest_fitness[current_best_idx] > gbest_fitness:
            gbest_fitness = pbest_fitness[current_best_idx]
            gbest_position = pbest_positions[current_best_idx].copy()

        diversity = swarm_diversity(positions, n_jobs)
        convergence_log.append({
            "iteration": it,
            "best_fitness": round(gbest_fitness, 4),
            "avg_fitness": round(fitness_values.mean(), 4),
            "diversity": round(diversity, 4),
            "inertia": round(inertia, 3),
        })

        if verbose and it % max(1, iterations // 10) == 0:
            print(f"[PSO] Iter {it:4d} | Best: {gbest_fitness:.2f} | "
                  f"Avg: {fitness_values.mean():.2f} | Diversity: {diversity:.3f} | "
                  f"w: {inertia:.3f}")

    best_position_discrete = decode_position(gbest_position, n_jobs)

    if verbose:
        print(f"\n[PSO] Selesai. Fitness terbaik akhir: {gbest_fitness:.2f}")

    return best_position_discrete, gbest_fitness, convergence_log


# ============================================================
# MODE STANDALONE (opsional, untuk testing modul ini sendiri)
# ============================================================
if __name__ == "__main__":
    from src.Preprocessing import (load_and_preprocess)
    from src.GeneticAlgorithm import solution_to_dataframe

    df_grad, df_job = load_and_preprocess(
        grad_path="../data/graduates_dummy.csv",
        job_path="../data/job_postings_dummy.csv",
        verbose=False,
    )

    np.random.seed(42)

    fitness_matrix = precompute_fitness_matrix(df_grad, df_job)
    best_position, best_fitness, log = run_pso(
        df_grad, df_job, iterations=500, fitness_matrix=fitness_matrix
    )

    df_result = solution_to_dataframe(best_position, df_grad, df_job, fitness_matrix)

    out_path = "../output/pso_result.csv"
    log_path = "../output/pso_convergence_log.csv"

    df_result.to_csv(out_path, index=False)
    pd.DataFrame(log).to_csv(log_path, index=False)

    print(f"\n[TEST] Hasil matching PSO disimpan: {out_path}")
    print(f"[TEST] Log konvergensi disimpan   : {log_path}")
    print(f"\n[TEST] Rata-rata fitness solusi terbaik: {best_fitness:.2f}")
    print(df_result.head())