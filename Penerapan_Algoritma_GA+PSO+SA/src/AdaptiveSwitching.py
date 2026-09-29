"""
src/adaptive_switching.py

Modul ADAPTIVE SWITCHING MECHANISM
Bagian dari "Hybrid Optimization Core" (Gambar 2 paper), yang menghubungkan
GA, PSO, dan SA berdasarkan tingkat KEBERAGAMAN (diversity) populasi/swarm
saat ini -- bukan berdasarkan iterasi tetap atau performa historis.

Aturan sesuai paper (Bagian "Research Novelty and Contributions" & Tabel 3):
    Diversity > 0.7   -> HIGH    -> pakai GA  (perlu eksplorasi global lagi)
    0.3 <= Diversity <= 0.7 -> MEDIUM -> pakai PSO (penyempurnaan lokal)
    Diversity < 0.3   -> LOW     -> pakai SA  (berisiko local optimum,
                                               perlu "guncangan" keluar)

Modul ini TIDAK menjalankan algoritma itu sendiri -- ia hanya
memutuskan ALGORITMA MANA yang harus dijalankan berikutnya, berdasarkan
kondisi diversity saat ini. Eksekusi algoritmanya dilakukan oleh
`hybrid_optimizer.py` yang memanggil `genetic_algorithm.py`,
`particle_swarm.py`, atau `simulated_annealing.py` sesuai keputusan modul ini.

Cara pakai singkat:
    from src.adaptive_switching import select_algorithm, classify_diversity

    state = classify_diversity(0.65)      # -> "MEDIUM"
    next_algo = select_algorithm(0.65)    # -> "PSO"
"""

import numpy as np
import pandas as pd


# ============================================================
# THRESHOLD DIVERSITY (sesuai paper)
# ============================================================
DIVERSITY_THRESHOLDS = {
    "high": 0.7,   # di atas ini -> populasi masih sangat beragam
    "low": 0.3,    # di bawah ini -> populasi sudah menyatu/seragam
}

# Pemetaan kondisi diversity -> algoritma yang dijalankan berikutnya
DIVERSITY_ALGORITHM_MAP = {
    "HIGH": "GA",
    "MEDIUM": "PSO",
    "LOW": "SA",
}


# ============================================================
# TAHAP 1: KLASIFIKASI KONDISI DIVERSITY
# ============================================================
def classify_diversity(diversity: float) -> str:
    """
    Mengklasifikasikan nilai diversity (0-1) menjadi kondisi:
    "HIGH" (>0.7), "MEDIUM" (0.3-0.7), atau "LOW" (<0.3).
    """
    if diversity > DIVERSITY_THRESHOLDS["high"]:
        return "HIGH"
    elif diversity < DIVERSITY_THRESHOLDS["low"]:
        return "LOW"
    else:
        return "MEDIUM"


# ============================================================
# TAHAP 2: PEMILIHAN ALGORITMA BERIKUTNYA
# ============================================================
def select_algorithm(diversity: float) -> str:
    """
    Menentukan algoritma yang harus dijalankan berikutnya berdasarkan
    kondisi diversity saat ini:
        HIGH   -> GA  (global exploration, populasi masih beragam)
        MEDIUM -> PSO (local exploitation, mulai menyempit)
        LOW    -> SA  (escape local optima, sudah terlalu seragam)
    """
    state = classify_diversity(diversity)
    return DIVERSITY_ALGORITHM_MAP[state]


# ============================================================
# TAHAP 3: DETEKSI STAGNASI
# (pelengkap keputusan switching -- dipakai bersama diversity oleh
#  hybrid_optimizer.py untuk memutuskan kapan perlu SA meski diversity
#  belum benar-benar rendah, mis. saat fitness tidak membaik sekian iterasi)
# ============================================================
def is_stagnant(fitness_history: list, patience: int = 20,
                 min_improvement: float = 1e-4) -> bool:
    """
    Mengecek apakah fitness terbaik tidak membaik secara signifikan
    dalam `patience` iterasi/generasi terakhir.
    """
    if len(fitness_history) < patience:
        return False

    recent = fitness_history[-patience:]
    improvement = max(recent) - recent[0]
    return improvement < min_improvement


# ============================================================
# TAHAP 4: PENCATATAN KEPUTUSAN SWITCHING (untuk log & analisis)
# ============================================================
def log_switch_decision(iteration: int, diversity: float, chosen_algorithm: str,
                         reason: str = "") -> dict:
    """
    Membuat satu entri log keputusan switching -- dipakai hybrid_optimizer.py
    untuk mencatat riwayat perpindahan antar algoritma sepanjang proses,
    berguna untuk analisis & visualisasi (mis. anotasi Gambar 4 pada paper).
    """
    return {
        "iteration": iteration,
        "diversity": round(diversity, 4),
        "diversity_state": classify_diversity(diversity),
        "chosen_algorithm": chosen_algorithm,
        "reason": reason,
    }


# ============================================================
# UTILITAS: RINGKASAN STATISTIK SWITCHING
# (berguna untuk laporan -- mis. "GA dipakai 35% dari total iterasi")
# ============================================================
def summarize_switch_log(switch_log: list) -> pd.DataFrame:
    """
    Merangkum riwayat switching menjadi tabel frekuensi & proporsi
    penggunaan tiap algoritma sepanjang proses hybrid optimization.
    """
    df_log = pd.DataFrame(switch_log)
    if df_log.empty:
        return pd.DataFrame(columns=["algorithm", "count", "percentage"])

    summary = (
        df_log["chosen_algorithm"]
        .value_counts()
        .rename_axis("algorithm")
        .reset_index(name="count")
    )
    summary["percentage"] = round(summary["count"] / summary["count"].sum() * 100, 1)
    return summary


# ============================================================
# MODE STANDALONE (opsional, untuk testing modul ini sendiri)
# ============================================================
if __name__ == "__main__":
    # --- Tes 1: klasifikasi & pemilihan algoritma pada beberapa nilai diversity
    test_diversities = [0.85, 0.72, 0.50, 0.31, 0.28, 0.10]

    print("[TEST] Klasifikasi & pemilihan algoritma:")
    for d in test_diversities:
        state = classify_diversity(d)
        algo = select_algorithm(d)
        print(f"  diversity={d:.2f} -> state={state:6s} -> algoritma={algo}")

    # --- Tes 2: simulasi log switching sepanjang beberapa iterasi
    print("\n[TEST] Simulasi log switching (data diversity sintetis):")
    np.random.seed(42)
    simulated_diversity = np.clip(
        np.concatenate([
            np.linspace(0.9, 0.4, 60),      # fase awal: diversity menurun (GA aktif)
            np.random.uniform(0.2, 0.4, 20),  # fase tengah: mulai rendah (PSO/SA)
            np.random.uniform(0.05, 0.25, 20),  # fase akhir: rendah (SA aktif)
        ]),
        0, 1
    )

    switch_log = []
    fitness_history = []
    current_fitness = 60.0

    for it, d in enumerate(simulated_diversity):
        algo = select_algorithm(d)
        # simulasi fitness naik sedikit tiap iterasi (utk demo is_stagnant saja)
        current_fitness += np.random.uniform(-0.1, 0.3)
        fitness_history.append(current_fitness)

        reason = "stagnant" if is_stagnant(fitness_history) else "diversity-based"
        switch_log.append(log_switch_decision(it, d, algo, reason))

    df_summary = summarize_switch_log(switch_log)
    print(df_summary)

    out_path = "/home/claude/project_graduate_matching/outputs/switch_log_demo.csv"
    pd.DataFrame(switch_log).to_csv(out_path, index=False)
    print(f"\n[TEST] Log switching demo disimpan: {out_path}")