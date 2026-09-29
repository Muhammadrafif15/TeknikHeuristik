"""
src/hybrid_optimizer.py

Modul HYBRID OPTIMIZATION CORE
Menggabungkan GA (global exploration), PSO (local exploitation), dan SA
(escape local optima) menjadi satu siklus optimasi tunggal, dikendalikan
oleh Adaptive Switching Mechanism (`adaptive_switching.py`) berdasarkan
tingkat diversity populasi -- inilah inti dari algoritma GA-PSO-SA hybrid
yang diusulkan pada paper.

=====================================================================
CATATAN PENTING TENTANG PENYATUAN REPRESENTASI (Design Note)
=====================================================================
GA & SA pada paper bekerja di ruang solusi DISKRIT (index job integer),
sedangkan PSO bekerja di ruang KONTINU (posisi float). Agar ketiganya bisa
"serah-terima" solusi secara mulus dalam satu siklus hybrid, modul ini
menyatukan representasi sbb:

    - "Populasi utama" (`positions`) disimpan sbg array KONTINU
      (pop_size x n_graduates), persis seperti representasi PSO.
    - Saat giliran GA aktif, operator GA (tournament selection, crossover,
      mutation dari `genetic_algorithm.py`) diterapkan LANGSUNG pada array
      kontinu ini (operator-operator tsb tidak peduli tipe data float/int,
      karena fitness selalu dihitung via decode_position() -> integer).
    - Saat giliran PSO aktif, update velocity & posisi standar PSO
      diterapkan pada `positions` yang sama.
    - Saat giliran SA aktif, satu solusi terbaik (gbest) didekode jadi
      diskrit, "diguncang" (neighbor search), lalu hasilnya disisipkan
      kembali ke populasi (menggantikan individu terlemah).

Penyederhanaan ini memungkinkan ketiga algoritma berbagi satu populasi
yang sama sepanjang proses, sesuai semangat "hybrid" pada paper, meski
detail implementasi teknisnya adalah interpretasi kami sendiri karena
paper tidak merinci mekanisme penyatuan representasi ini secara eksplisit.
=====================================================================

Cara pakai singkat:
    from src.hybrid_optimizer import run_hybrid_optimization

    best_solution, best_fitness, conv_log, switch_log = run_hybrid_optimization(
        df_grad, df_job, max_iterations=500
    )
"""

import time
import numpy as np
import pandas as pd

from src.GeneticAlgorithm import (
    precompute_fitness_matrix, tournament_selection, crossover, mutate,
    get_elites, population_diversity, GA_PARAMS,
)
from src.ParticleSwarm import (
    initialize_swarm, evaluate_swarm, decode_position, update_velocity,
    update_position, linear_inertia_decay, PSO_PARAMS,
)
from src.SimulatedAnnealing import (
    evaluate_solution, generate_neighbor, acceptance_probability, cool_down,
    SA_PARAMS,
)
from src.AdaptiveSwitching import select_algorithm, classify_diversity


# ============================================================
# PARAMETER DEFAULT HYBRID (sesuai Tabel 3 paper)
# ============================================================
HYBRID_PARAMS = {
    "max_iterations": 500,
    "convergence_threshold": 1e-6,
    "convergence_patience": 30,   # jumlah iterasi berturut2 tanpa perbaikan
                                   # signifikan sebelum dianggap konvergen
                                   # (tidak dirinci eksplisit di paper --
                                   # ditambahkan agar proses bisa berhenti lebih awal)
}


# ============================================================
# METRIK DIVERSITY (PERBAIKAN)
# ============================================================
def _decoded_population_diversity(positions: np.ndarray, n_jobs: int) -> float:
    """
    Menghitung diversity berbasis RASIO KEUNIKAN NILAI pada populasi yang
    sudah didekode ke bentuk diskrit (bukan std/range dari posisi kontinu).

    Kenapa diganti dari `swarm_diversity()` (berbasis std)?
    Untuk populasi yang diinisialisasi acak seragam (uniform random) di
    rentang [0, n_jobs), nilai std/range-nya SECARA MATEMATIS sudah berada
    di sekitar 0.28-0.29 -- yaitu SUDAH di bawah threshold LOW (<0.3) sejak
    populasi awal. Akibatnya sistem langsung mengira populasi "menyatu"
    padahal masih 100% acak, sehingga SA selalu terpilih dan GA/PSO tidak
    pernah kebagian giliran.

    Metrik berbasis rasio keunikan nilai (dipinjam dari
    `genetic_algorithm.population_diversity()`) tidak punya bias ini: saat
    populasi masih acak, hampir semua nilai pada tiap gen berbeda-beda
    (rasio keunikan tinggi -> diversity tinggi -> HIGH -> GA aktif), dan
    baru turun mendekati 0 saat populasi benar-benar mulai seragam.

    PERBAIKAN TAMBAHAN (bug kedua): `population_diversity()` versi asli
    membagi jumlah nilai unik dengan `pop_size`. Jika `pop_size > n_jobs`
    (mis. 50 individu tapi hanya 25 pilihan job), jumlah nilai unik TIDAK
    PERNAH bisa melebihi `n_jobs`, sehingga diversity maksimum yang bisa
    dicapai hanya `n_jobs / pop_size` (mis. 25/50 = 0.5) -- kondisi "HIGH"
    (>0.7) jadi MUSTAHIL tercapai meski populasi 100% acak, dan GA tidak
    pernah aktif. Perbaikannya: normalisasi dibagi `min(pop_size, n_jobs)`
    (batas keunikan yang SECARA REALISTIS bisa dicapai), bukan `pop_size` saja.
    """
    decoded_population = np.array([decode_position(ind, n_jobs) for ind in positions])
    pop_size = decoded_population.shape[0]
    max_possible_unique = min(pop_size, n_jobs)

    unique_ratios = [
        len(np.unique(decoded_population[:, col])) / max_possible_unique
        for col in range(decoded_population.shape[1])
    ]
    return float(np.mean(unique_ratios))


# ============================================================
# TAHAP GA: SATU LANGKAH (generasi) GA pada populasi bersama
# ============================================================
def _ga_step(positions: np.ndarray, fitness_values: np.ndarray, n_jobs: int) -> np.ndarray:
    """Menjalankan satu generasi GA (elitism + tournament + crossover + mutasi)
    langsung pada array populasi kontinu."""
    pop_size = positions.shape[0]

    next_population = list(get_elites(positions, fitness_values, GA_PARAMS["elitism_rate"]))

    while len(next_population) < pop_size:
        parent1 = tournament_selection(positions, fitness_values, GA_PARAMS["tournament_k"])
        parent2 = tournament_selection(positions, fitness_values, GA_PARAMS["tournament_k"])

        child1, child2 = crossover(parent1, parent2, GA_PARAMS["crossover_rate"])
        child1 = mutate(child1, GA_PARAMS["mutation_rate"], n_jobs)
        child2 = mutate(child2, GA_PARAMS["mutation_rate"], n_jobs)

        next_population.append(child1)
        if len(next_population) < pop_size:
            next_population.append(child2)

    return np.array(next_population[:pop_size], dtype=float)


# ============================================================
# TAHAP PSO: SATU LANGKAH (iterasi) PSO pada populasi bersama
# ============================================================
def _pso_step(positions: np.ndarray, velocities: np.ndarray,
               pbest_positions: np.ndarray, gbest_position: np.ndarray,
               iteration: int, max_iterations: int, n_jobs: int) -> tuple:
    """Menjalankan satu iterasi PSO (update velocity & posisi)."""
    max_velocity = PSO_PARAMS["velocity_clamp_ratio"] * n_jobs
    inertia = linear_inertia_decay(iteration, max_iterations,
                                    PSO_PARAMS["inertia_start"], PSO_PARAMS["inertia_end"])

    velocities = update_velocity(
        velocities, positions, pbest_positions, gbest_position,
        inertia, PSO_PARAMS["c1"], PSO_PARAMS["c2"], max_velocity
    )
    positions = update_position(positions, velocities, n_jobs)
    return positions, velocities, inertia


# ============================================================
# TAHAP SA: SATU LANGKAH SA, hasilnya disisipkan ke populasi
# ============================================================
def _sa_step(positions: np.ndarray, fitness_values: np.ndarray,
              sa_state: dict, fitness_matrix: np.ndarray, n_jobs: int) -> tuple:
    """
    Menjalankan satu langkah SA berbasis solusi terbaik saat ini (gbest),
    lalu menyisipkan hasilnya ke individu terlemah dalam populasi (agar
    solusi SA ikut "diwariskan" ke fase GA/PSO berikutnya).

    sa_state (dict, persist antar-pemanggilan) berisi:
        current_solution, current_fitness, temperature, stagnation_counter
    """
    current_solution = sa_state["current_solution"]
    current_fitness = sa_state["current_fitness"]
    temperature = sa_state["temperature"]
    stagnation_counter = sa_state["stagnation_counter"]

    neighbor = generate_neighbor(current_solution, n_jobs, SA_PARAMS["genes_to_perturb"])
    neighbor_fitness = evaluate_solution(neighbor, fitness_matrix)

    delta = neighbor_fitness - current_fitness
    prob_accept = acceptance_probability(delta, temperature)
    accepted = np.random.rand() < prob_accept

    if accepted:
        current_solution = neighbor
        current_fitness = neighbor_fitness

    improved = current_fitness > sa_state["best_fitness_seen"]
    if improved:
        stagnation_counter = 0
    else:
        stagnation_counter += 1

    reheated = False
    if stagnation_counter >= SA_PARAMS["reheat_threshold"]:
        temperature = SA_PARAMS["reheat_temperature"]
        stagnation_counter = 0
        reheated = True
    else:
        temperature = max(cool_down(temperature, SA_PARAMS["cooling_rate"]),
                           SA_PARAMS["min_temperature"])

    # --- Sisipkan solusi SA ke individu terlemah dalam populasi
    worst_idx = np.argmin(fitness_values)
    positions[worst_idx] = current_solution.astype(float)

    sa_state.update({
        "current_solution": current_solution,
        "current_fitness": current_fitness,
        "temperature": temperature,
        "stagnation_counter": stagnation_counter,
        "best_fitness_seen": max(sa_state["best_fitness_seen"], current_fitness),
    })

    return positions, sa_state, reheated


# ============================================================
# FUNGSI UTAMA: HYBRID OPTIMIZATION (GA + PSO + SA + Adaptive Switching)
# ============================================================
def run_hybrid_optimization(
    df_grad: pd.DataFrame,
    df_job: pd.DataFrame,
    max_iterations: int = None,
    params: dict = None,
    verbose: bool = True,
) -> tuple:
    """
    Menjalankan siklus optimasi hybrid GA-PSO-SA dengan adaptive
    diversity-based switching, sesuai Gambar 2 & Tabel 3 paper.

    Returns
    -------
    best_solution : np.ndarray
        Assignment terbaik (index job untuk tiap graduate) sepanjang proses.
    best_fitness : float
        Nilai fitness dari best_solution.
    convergence_log : list of dict
        Riwayat {iteration, diversity, algorithm, best_fitness, avg_fitness}.
    switch_log : list of dict
        Riwayat keputusan switching (dari `adaptive_switching.py`).
    """
    p = {**HYBRID_PARAMS, **(params or {})}
    if max_iterations is not None:
        p["max_iterations"] = max_iterations

    n_graduates = len(df_grad)
    n_jobs = len(df_job)

    if verbose:
        print("[HYBRID] Menghitung fitness matrix (sekali saja)...")
    fitness_matrix = precompute_fitness_matrix(df_grad, df_job)

    # --- Inisialisasi populasi/swarm bersama (representasi kontinu, ala PSO)
    pop_size = PSO_PARAMS["swarm_size"]
    positions, velocities = initialize_swarm(pop_size, n_graduates, n_jobs)

    fitness_values = evaluate_swarm(positions, fitness_matrix, n_jobs)
    pbest_positions = positions.copy()
    pbest_fitness = fitness_values.copy()

    gbest_idx = np.argmax(pbest_fitness)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]

    # --- Inisialisasi state SA (persisten antar-iterasi)
    sa_state = {
        "current_solution": decode_position(gbest_position, n_jobs),
        "current_fitness": gbest_fitness,
        "temperature": SA_PARAMS["initial_temperature"],
        "stagnation_counter": 0,
        "best_fitness_seen": gbest_fitness,
    }

    convergence_log = []
    switch_log = []
    no_improve_count = 0
    prev_best = gbest_fitness

    start_time = time.perf_counter()

    for it in range(p["max_iterations"]):
        diversity = _decoded_population_diversity(positions, n_jobs)
        algo = select_algorithm(diversity)

        if algo == "GA":
            positions = _ga_step(positions, fitness_values, n_jobs)
        elif algo == "PSO":
            positions, velocities, _ = _pso_step(
                positions, velocities, pbest_positions, gbest_position,
                it, p["max_iterations"], n_jobs
            )
        else:  # "SA"
            positions, sa_state, _ = _sa_step(
                positions, fitness_values, sa_state, fitness_matrix, n_jobs
            )

        # --- Evaluasi ulang & update pbest/gbest (berlaku utk semua kasus)
        fitness_values = evaluate_swarm(positions, fitness_matrix, n_jobs)
        improved_mask = fitness_values > pbest_fitness
        pbest_positions[improved_mask] = positions[improved_mask]
        pbest_fitness[improved_mask] = fitness_values[improved_mask]

        current_best_idx = np.argmax(pbest_fitness)
        if pbest_fitness[current_best_idx] > gbest_fitness:
            gbest_fitness = pbest_fitness[current_best_idx]
            gbest_position = pbest_positions[current_best_idx].copy()

        # --- Cek konvergensi
        if (gbest_fitness - prev_best) < p["convergence_threshold"]:
            no_improve_count += 1
        else:
            no_improve_count = 0
        prev_best = gbest_fitness

        switch_log.append({
            "iteration": it,
            "diversity": round(diversity, 4),
            "diversity_state": classify_diversity(diversity),
            "chosen_algorithm": algo,
        })
        convergence_log.append({
            "iteration": it,
            "diversity": round(diversity, 4),
            "algorithm": algo,
            "best_fitness": round(gbest_fitness, 4),
            "avg_fitness": round(fitness_values.mean(), 4),
        })

        if verbose and it % max(1, p["max_iterations"] // 10) == 0:
            print(f"[HYBRID] Iter {it:4d} | Algo: {algo:3s} | "
                  f"Diversity: {diversity:.3f} | Best: {gbest_fitness:.2f} | "
                  f"Avg: {fitness_values.mean():.2f}")

        if no_improve_count >= p["convergence_patience"]:
            if verbose:
                print(f"\n[HYBRID] Konvergen pada iterasi {it} "
                      f"(tidak ada perbaikan signifikan selama "
                      f"{p['convergence_patience']} iterasi berturut-turut)")
            break

    elapsed_time = time.perf_counter() - start_time
    best_solution = decode_position(gbest_position, n_jobs)

    if verbose:
        print(f"\n[HYBRID] Selesai dalam {elapsed_time:.2f} detik "
              f"({it + 1} iterasi). Fitness terbaik: {gbest_fitness:.2f}")

    return best_solution, gbest_fitness, convergence_log, switch_log


# ============================================================
# MODE STANDALONE (opsional, untuk testing modul ini sendiri)
# ============================================================
if __name__ == "__main__":
    from src.Preprocessing import load_and_preprocess
    from src.GeneticAlgorithm import solution_to_dataframe
    from src.AdaptiveSwitching import (summarize_switch_log)

    df_grad, df_job = load_and_preprocess(
        grad_path="../data/graduates_dummy.csv",
        job_path="../data/job_postings_dummy.csv",
        verbose=False,
    )

    np.random.seed(42)

    best_solution, best_fitness, conv_log, switch_log = run_hybrid_optimization(
        df_grad, df_job, max_iterations=500
    )

    fitness_matrix = precompute_fitness_matrix(df_grad, df_job)
    df_result = solution_to_dataframe(best_solution, df_grad, df_job, fitness_matrix)

    result_path = "../output/hybrid_result.csv"
    conv_log_path = "../output/hybrid_convergence_log.csv"
    switch_log_path = "../output/hybrid_switch_log.csv"

    df_result.to_csv(result_path, index=False)
    pd.DataFrame(conv_log).to_csv(conv_log_path, index=False)
    pd.DataFrame(switch_log).to_csv(switch_log_path, index=False)

    print(f"\n[TEST] Hasil matching hybrid disimpan : {result_path}")
    print(f"[TEST] Log konvergensi disimpan        : {conv_log_path}")
    print(f"[TEST] Log switching disimpan          : {switch_log_path}")

    print("\n[TEST] Ringkasan penggunaan algoritma sepanjang proses:")
    print(summarize_switch_log(switch_log))

    print(f"\n[TEST] Rata-rata fitness solusi terbaik: {best_fitness:.2f}")
    print(df_result.head())