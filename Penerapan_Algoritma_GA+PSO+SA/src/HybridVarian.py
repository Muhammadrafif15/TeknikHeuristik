"""
src/hybrid_variants.py

EKSPERIMEN: TIGA TAFSIRAN MEKANISME PERALIHAN GA-PSO-SA

Paper menyebut dua hal yang tidak didamaikan secara eksplisit:
    (1) Peralihan algoritma berbasis DIVERSITY (>0.7 GA, 0.3-0.7 PSO, <0.3 SA)
    (2) Narasi hasil (Bagian 3.1, Gambar 4) menyebut TIGA FASE dengan rentang
        iterasi tetap: GA 0-100, PSO 100-300, SA 300-500

Modul ini menguji tiga tafsiran secara empiris, semua dibatasi 500 iterasi:

    VARIAN 1 (pure_diversity)   : HANYA aturan diversity, dicek TIAP iterasi,
                                   tanpa fase GA yang dipaksa di awal.
    VARIAN 2 (fixed_schedule)   : HANYA jadwal iterasi tetap (GA 0-100,
                                   PSO 100-300, SA 300-500), diversity
                                   dicatat tapi TIDAK memengaruhi keputusan.
    VARIAN 3 (threshold_or_cap) : Sekuensial GA -> PSO -> SA (satu arah,
                                   TIDAK bolak-balik). Tiap fase berpindah ke
                                   fase berikutnya begitu syarat diversity-nya
                                   terpenuhi, ATAU begitu batas iterasi fase
                                   tsb tercapai -- mana yang lebih dulu.

Cara pakai singkat:
    from src.hybrid_variants import run_variant_1, run_variant_2, run_variant_3
"""

import numpy as np
import pandas as pd

from src.GeneticAlgorithm import precompute_fitness_matrix, GA_PARAMS
from src.ParticleSwarm import initialize_swarm, evaluate_swarm, decode_position, PSO_PARAMS
from src.SimulatedAnnealing import SA_PARAMS
from src.AdaptiveSwitching import select_algorithm, classify_diversity
from src.HybridOptimizer import (
    _decoded_population_diversity, _ga_step, _pso_step, _sa_step,
)


# ============================================================
# UTILITAS BERSAMA: INISIALISASI STATE (dipakai ketiga varian)
# ============================================================
def _init_state(df_grad, df_job, fitness_matrix):
    n_graduates = len(df_grad)
    n_jobs = len(df_job)
    pop_size = GA_PARAMS["population_size"]  # 100, sesuai Tabel 3

    positions, velocities = initialize_swarm(pop_size, n_graduates, n_jobs)
    fitness_values = evaluate_swarm(positions, fitness_matrix, n_jobs)
    pbest_positions = positions.copy()
    pbest_fitness = fitness_values.copy()

    gbest_idx = np.argmax(pbest_fitness)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]

    sa_state = {
        "current_solution": decode_position(gbest_position, n_jobs),
        "current_fitness": gbest_fitness,
        "temperature": SA_PARAMS["initial_temperature"],
        "stagnation_counter": 0,
        "best_fitness_seen": gbest_fitness,
    }

    return {
        "positions": positions, "velocities": velocities,
        "fitness_values": fitness_values,
        "pbest_positions": pbest_positions, "pbest_fitness": pbest_fitness,
        "gbest_position": gbest_position, "gbest_fitness": gbest_fitness,
        "sa_state": sa_state, "n_graduates": n_graduates, "n_jobs": n_jobs,
    }


def _run_one_step(state, algo, it, max_iterations, fitness_matrix,
                   sa_reheat_threshold, sa_refresh_fraction):
    """Menjalankan satu langkah algoritma `algo`, meng-update state di tempat,
    lalu evaluasi ulang & update pbest/gbest. Dipakai bersama oleh ketiga varian."""
    n_jobs = state["n_jobs"]

    if algo == "GA":
        state["positions"] = _ga_step(state["positions"], state["fitness_values"], n_jobs)
    elif algo == "PSO":
        state["positions"], state["velocities"], _ = _pso_step(
            state["positions"], state["velocities"],
            state["pbest_positions"], state["gbest_position"],
            it, max_iterations, n_jobs
        )
    else:  # "SA"
        state["positions"], state["sa_state"], reheated = _sa_step(
            state["positions"], state["fitness_values"], state["sa_state"],
            fitness_matrix, n_jobs, reheat_threshold=sa_reheat_threshold,
        )
        if reheated:
            n_refresh = max(1, int(sa_refresh_fraction * state["positions"].shape[0]))
            worst_idx = np.argsort(state["fitness_values"])[:n_refresh]
            state["positions"][worst_idx] = np.random.uniform(
                0, n_jobs, size=(n_refresh, state["n_graduates"])
            )

    state["fitness_values"] = evaluate_swarm(state["positions"], fitness_matrix, n_jobs)
    improved = state["fitness_values"] > state["pbest_fitness"]
    state["pbest_positions"][improved] = state["positions"][improved]
    state["pbest_fitness"][improved] = state["fitness_values"][improved]

    best_idx = np.argmax(state["pbest_fitness"])
    if state["pbest_fitness"][best_idx] > state["gbest_fitness"]:
        state["gbest_fitness"] = state["pbest_fitness"][best_idx]
        state["gbest_position"] = state["pbest_positions"][best_idx].copy()

    return state


# ============================================================
# VARIAN 1: MURNI DIVERSITY (dicek TIAP iterasi, tanpa fase dipaksa)
# ============================================================
def run_variant_1(df_grad, df_job, max_iterations=500,
                   sa_reheat_threshold=50, sa_refresh_fraction=0.4,
                   fitness_matrix=None, verbose=False):
    n_jobs = len(df_job)
    if fitness_matrix is None:
        fitness_matrix = precompute_fitness_matrix(df_grad, df_job)
    state = _init_state(df_grad, df_job, fitness_matrix)

    log = []
    for it in range(max_iterations):
        diversity = _decoded_population_diversity(state["positions"], n_jobs)
        algo = select_algorithm(diversity)  # MURNI diversity, tiap iterasi

        state = _run_one_step(state, algo, it, max_iterations, fitness_matrix,
                               sa_reheat_threshold, sa_refresh_fraction)

        log.append({"iteration": it, "algorithm": algo, "diversity": round(diversity, 4),
                     "best_fitness": round(state["gbest_fitness"], 4)})
        if verbose and it % 50 == 0:
            print(f"[V1] it={it:4d} algo={algo:3s} div={diversity:.3f} "
                  f"best={state['gbest_fitness']:.2f}")

    best_solution = decode_position(state["gbest_position"], n_jobs)
    return best_solution, state["gbest_fitness"], pd.DataFrame(log)


# ============================================================
# VARIAN 2: MURNI JADWAL TETAP (GA 0-100, PSO 100-300, SA 300-500)
# ============================================================
def run_variant_2(df_grad, df_job, max_iterations=500,
                   ga_iters=100, pso_iters=200, sa_iters=200,
                   sa_reheat_threshold=50, sa_refresh_fraction=0.4,
                   fitness_matrix=None, verbose=False):
    assert ga_iters + pso_iters + sa_iters == max_iterations, \
        "Jadwal fase harus berjumlah tepat max_iterations"

    n_jobs = len(df_job)
    if fitness_matrix is None:
        fitness_matrix = precompute_fitness_matrix(df_grad, df_job)
    state = _init_state(df_grad, df_job, fitness_matrix)

    log = []
    for it in range(max_iterations):
        if it < ga_iters:
            algo = "GA"
        elif it < ga_iters + pso_iters:
            algo = "PSO"
        else:
            algo = "SA"

        # diversity dihitung HANYA untuk dicatat/dibandingkan, TIDAK
        # memengaruhi keputusan algoritma pada varian ini
        diversity = _decoded_population_diversity(state["positions"], n_jobs)

        state = _run_one_step(state, algo, it, max_iterations, fitness_matrix,
                               sa_reheat_threshold, sa_refresh_fraction)

        log.append({"iteration": it, "algorithm": algo, "diversity": round(diversity, 4),
                     "best_fitness": round(state["gbest_fitness"], 4)})
        if verbose and it % 50 == 0:
            print(f"[V2] it={it:4d} algo={algo:3s} div={diversity:.3f} "
                  f"best={state['gbest_fitness']:.2f}")

    best_solution = decode_position(state["gbest_position"], n_jobs)
    return best_solution, state["gbest_fitness"], pd.DataFrame(log)


# ============================================================
# VARIAN 3: THRESHOLD-OR-CAP (sekuensial GA -> PSO -> SA, satu arah)
# Pindah fase begitu syarat diversity terpenuhi ATAU batas iterasi fase
# tercapai -- mana yang lebih dulu.
# ============================================================
def run_variant_3(df_grad, df_job, max_iterations=500,
                   ga_max_iters=100, pso_max_iters=200,
                   sa_reheat_threshold=50, sa_refresh_fraction=0.4,
                   fitness_matrix=None, verbose=False):
    n_jobs = len(df_job)
    if fitness_matrix is None:
        fitness_matrix = precompute_fitness_matrix(df_grad, df_job)
    state = _init_state(df_grad, df_job, fitness_matrix)

    log = []
    it = 0

    # --- FASE 1: GA, sampai diversity turun dari HIGH (<=0.7) atau capai cap
    phase_start = it
    while it < max_iterations and (it - phase_start) < ga_max_iters:
        diversity = _decoded_population_diversity(state["positions"], n_jobs)
        if classify_diversity(diversity) != "HIGH":
            break  # syarat exit fase terpenuhi lebih awal
        state = _run_one_step(state, "GA", it, max_iterations, fitness_matrix,
                               sa_reheat_threshold, sa_refresh_fraction)
        log.append({"iteration": it, "algorithm": "GA", "diversity": round(diversity, 4),
                     "best_fitness": round(state["gbest_fitness"], 4)})
        it += 1
    exit_reason_ga = "diversity_exit" if (it - phase_start) < ga_max_iters else "cap_reached"

    # --- FASE 2: PSO, sampai diversity turun ke LOW (<0.3) atau capai cap
    phase_start = it
    while it < max_iterations and (it - phase_start) < pso_max_iters:
        diversity = _decoded_population_diversity(state["positions"], n_jobs)
        if classify_diversity(diversity) == "LOW":
            break
        state = _run_one_step(state, "PSO", it, max_iterations, fitness_matrix,
                               sa_reheat_threshold, sa_refresh_fraction)
        log.append({"iteration": it, "algorithm": "PSO", "diversity": round(diversity, 4),
                     "best_fitness": round(state["gbest_fitness"], 4)})
        it += 1
    exit_reason_pso = "diversity_exit" if (it - phase_start) < pso_max_iters else "cap_reached"

    # --- FASE 3: SA, menghabiskan sisa budget (tidak ada fase berikutnya)
    while it < max_iterations:
        diversity = _decoded_population_diversity(state["positions"], n_jobs)
        state = _run_one_step(state, "SA", it, max_iterations, fitness_matrix,
                               sa_reheat_threshold, sa_refresh_fraction)
        log.append({"iteration": it, "algorithm": "SA", "diversity": round(diversity, 4),
                     "best_fitness": round(state["gbest_fitness"], 4)})
        it += 1

    if verbose:
        print(f"[V3] Fase GA berakhir krn: {exit_reason_ga} (di iterasi {phase_start})")
        print(f"[V3] Fase PSO berakhir krn: {exit_reason_pso}")

    best_solution = decode_position(state["gbest_position"], n_jobs)
    return best_solution, state["gbest_fitness"], pd.DataFrame(log)


# ============================================================
# MODE STANDALONE: JALANKAN & BANDINGKAN KETIGA VARIAN
# ============================================================
if __name__ == "__main__":
    from src.Preprocessing import load_and_preprocess
    from src.GeneticAlgorithm import solution_to_dataframe

    df_grad, df_job = load_and_preprocess(
        grad_path="../data/graduates_dummy.csv",
        job_path="../data/job_postings_dummy.csv",
        verbose=False,
    )

    fitness_matrix = precompute_fitness_matrix(df_grad, df_job)

    print("=" * 60)
    print("VARIAN 1: MURNI DIVERSITY (tiap iterasi, tanpa fase dipaksa)")
    print("=" * 60)
    np.random.seed(42)
    sol1, fit1, log1 = run_variant_1(df_grad, df_job, fitness_matrix=fitness_matrix, verbose=True)
    print(f"Fitness akhir: {fit1:.2f}")
    print("Distribusi algoritma:")
    print(log1["algorithm"].value_counts(normalize=True).round(3) * 100)

    print("\n" + "=" * 60)
    print("VARIAN 2: MURNI JADWAL TETAP (GA 0-100, PSO 100-300, SA 300-500)")
    print("=" * 60)
    np.random.seed(42)
    sol2, fit2, log2 = run_variant_2(df_grad, df_job, fitness_matrix=fitness_matrix, verbose=True)
    print(f"Fitness akhir: {fit2:.2f}")

    print("\n" + "=" * 60)
    print("VARIAN 3: THRESHOLD-OR-CAP (sekuensial, exit dini atau cap)")
    print("=" * 60)
    np.random.seed(42)
    sol3, fit3, log3 = run_variant_3(df_grad, df_job, fitness_matrix=fitness_matrix, verbose=True)
    print(f"Fitness akhir: {fit3:.2f}")
    print("Distribusi algoritma:")
    print(log3["algorithm"].value_counts(normalize=True).round(3) * 100)

    # --- Simpan hasil
    out_dir = "../output"
    log1.to_csv(f"{out_dir}/variant1_pure_diversity_log.csv", index=False)
    log2.to_csv(f"{out_dir}/variant2_fixed_schedule_log.csv", index=False)
    log3.to_csv(f"{out_dir}/variant3_threshold_or_cap_log.csv", index=False)

    df_comparison = pd.DataFrame([
        {"variant": "1. Pure diversity", "fitness": round(fit1, 2), "iterations_run": len(log1)},
        {"variant": "2. Fixed schedule", "fitness": round(fit2, 2), "iterations_run": len(log2)},
        {"variant": "3. Threshold-or-cap", "fitness": round(fit3, 2), "iterations_run": len(log3)},
    ])
    df_comparison.to_csv(f"{out_dir}/variants_comparison.csv", index=False)

    print("\n" + "=" * 60)
    print("RINGKASAN PERBANDINGAN 3 VARIAN")
    print("=" * 60)
    print(df_comparison.to_string(index=False))