"""
src/genetic_algorithm.py

Modul GENETIC ALGORITHM (GA) -- "Global Exploration"
Komponen pertama dari Hybrid Optimization Core (Gambar 2 paper).

Representasi Solusi (Chromosome Encoding):
    Satu individu/kromosom = array integer sepanjang jumlah graduate,
    dimana gen ke-i berisi INDEX job yang di-assign ke graduate ke-i.

    Contoh (5 graduate, 3 job tersedia):
        individu = [2, 0, 0, 1, 2]
        artinya : G001->J_idx2, G002->J_idx0, G003->J_idx0,
                  G004->J_idx1, G005->J_idx2

Parameter GA sesuai Tabel 3 pada paper:
    Population Size   : 100 individu
    Selection Method  : Tournament (k=3)
    Crossover Rate    : 0.85 (adaptive -> disederhanakan jadi konstan di sini)
    Mutation Rate     : 0.05 (self-adaptive -> disederhanakan jadi konstan)
    Elitism Rate      : 0.10

Cara pakai singkat:
    from src.genetic_algorithm import run_ga

    best_individual, best_fitness, log = run_ga(
        df_grad, df_job, generations=100
    )
"""

import numpy as np
import pandas as pd

from src.FitnessFunction import calculate_overall_fitness


# ============================================================
# PARAMETER DEFAULT GA (sesuai Tabel 3 paper)
# ============================================================
GA_PARAMS = {
    "population_size": 100,
    "tournament_k": 3,
    "crossover_rate": 0.85,
    "mutation_rate": 0.05,
    "elitism_rate": 0.10,
}


# ============================================================
# TAHAP 0: PRECOMPUTE FITNESS MATRIX
# (agar GA tidak menghitung ulang cosine similarity / euclidean distance
#  setiap kali mengevaluasi individu -- cukup dihitung SEKALI di awal)
# ============================================================
def precompute_fitness_matrix(df_grad: pd.DataFrame, df_job: pd.DataFrame) -> np.ndarray:
    """
    Menghasilkan matriks fitness berukuran (n_graduates x n_jobs), dimana
    fitness_matrix[i][j] = skor F(x) antara graduate ke-i dan job ke-j.

    Matriks inilah yang dipakai berulang kali oleh GA/PSO/SA sebagai
    "lookup table", bukan menghitung ulang fitness function setiap generasi.
    """
    n_grad = len(df_grad)
    n_job = len(df_job)
    fitness_matrix = np.zeros((n_grad, n_job))

    for i in range(n_grad):
        grad_row = df_grad.iloc[i]
        for j in range(n_job):
            job_row = df_job.iloc[j]
            score, _ = calculate_overall_fitness(grad_row, job_row)
            fitness_matrix[i, j] = score

    return fitness_matrix


# ============================================================
# TAHAP 1: INISIALISASI POPULASI
# ============================================================
def initialize_population(pop_size: int, n_graduates: int, n_jobs: int) -> np.ndarray:
    """
    Membuat populasi awal secara acak.
    Return: array berukuran (pop_size, n_graduates), tiap elemen = index job.
    """
    return np.random.randint(0, n_jobs, size=(pop_size, n_graduates))


# ============================================================
# TAHAP 2: EVALUASI FITNESS POPULASI
# ============================================================
def evaluate_population(population: np.ndarray, fitness_matrix: np.ndarray) -> np.ndarray:
    """
    Menghitung fitness tiap individu = rata-rata skor F(x) seluruh graduate
    terhadap job yang di-assign padanya (lookup langsung ke fitness_matrix).

    Return: array 1D berisi nilai fitness tiap individu dalam populasi.
    """
    n_graduates = population.shape[1]
    grad_indices = np.arange(n_graduates)

    fitness_values = np.array([
        fitness_matrix[grad_indices, individual].mean()
        for individual in population
    ])
    return fitness_values


# ============================================================
# TAHAP 3: SELEKSI -> Tournament Selection (k=3)
# ============================================================
def tournament_selection(population: np.ndarray, fitness_values: np.ndarray,
                          k: int = 3) -> np.ndarray:
    """
    Memilih satu individu pemenang dari k individu yang diundi secara acak
    (yang punya fitness tertinggi di antara k tersebut yang menang).
    """
    pop_size = population.shape[0]
    contender_idx = np.random.choice(pop_size, size=k, replace=False)
    winner_idx = contender_idx[np.argmax(fitness_values[contender_idx])]
    return population[winner_idx].copy()


# ============================================================
# TAHAP 4: CROSSOVER -> Single-Point Crossover
# ============================================================
def crossover(parent1: np.ndarray, parent2: np.ndarray, crossover_rate: float) -> tuple:
    """
    Dengan probabilitas `crossover_rate`, tukar sebagian gen antara 2 parent
    di satu titik potong acak. Jika tidak, anak = salinan parent apa adanya.
    """
    if np.random.rand() > crossover_rate:
        return parent1.copy(), parent2.copy()

    n_genes = len(parent1)
    point = np.random.randint(1, n_genes)  # titik potong, hindari 0/ujung penuh

    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


# ============================================================
# TAHAP 5: MUTATION -> Random Reset Mutation
# ============================================================
def mutate(individual: np.ndarray, mutation_rate: float, n_jobs: int) -> np.ndarray:
    """
    Setiap gen punya peluang `mutation_rate` untuk diganti dengan job index
    acak yang baru (mensimulasikan mutasi pada GA klasik).
    """
    individual = individual.copy()
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = np.random.randint(0, n_jobs)
    return individual


# ============================================================
# TAHAP 6: ELITISM
# ============================================================
def get_elites(population: np.ndarray, fitness_values: np.ndarray,
                elitism_rate: float) -> np.ndarray:
    """
    Mengambil individu-individu terbaik (elite) untuk otomatis lolos ke
    generasi berikutnya tanpa proses seleksi/crossover/mutasi.
    """
    n_elites = max(1, int(len(population) * elitism_rate))
    elite_idx = np.argsort(fitness_values)[-n_elites:]
    return population[elite_idx].copy()


# ============================================================
# FUNGSI UTAMA: MENJALANKAN GA (satu siklus penuh evolusi)
# ============================================================
def run_ga(
    df_grad: pd.DataFrame,
    df_job: pd.DataFrame,
    generations: int = 100,
    params: dict = None,
    fitness_matrix: np.ndarray = None,
    verbose: bool = True,
) -> tuple:
    """
    Menjalankan Genetic Algorithm murni untuk mencari kombinasi assignment
    graduate->job terbaik.

    Parameters
    ----------
    df_grad, df_job : pandas.DataFrame
        Hasil dari `data_preprocessing.load_and_preprocess()`.
    generations : int
        Jumlah generasi/iterasi GA dijalankan.
    params : dict, optional
        Override parameter GA_PARAMS.
    fitness_matrix : np.ndarray, optional
        Jika sudah dihitung sebelumnya (mis. dipakai bersama PSO/SA dalam
        hybrid), bisa dioper langsung agar tidak dihitung ulang.
    verbose : bool
        Cetak progres tiap beberapa generasi.

    Returns
    -------
    best_individual : np.ndarray
        Assignment terbaik (index job untuk tiap graduate).
    best_fitness : float
        Nilai fitness (rata-rata F(x)) dari best_individual.
    convergence_log : list of dict
        Riwayat {generation, best_fitness, avg_fitness, diversity} tiap generasi
        -- dipakai nanti untuk membuat grafik konvergensi (spt Gambar 4 paper)
        dan sebagai input mekanisme adaptive switching.
    """
    p = {**GA_PARAMS, **(params or {})}
    n_graduates = len(df_grad)
    n_jobs = len(df_job)

    if fitness_matrix is None:
        if verbose:
            print("[GA] Menghitung fitness matrix (sekali saja)...")
        fitness_matrix = precompute_fitness_matrix(df_grad, df_job)

    population = initialize_population(p["population_size"], n_graduates, n_jobs)
    convergence_log = []

    for gen in range(generations):
        fitness_values = evaluate_population(population, fitness_matrix)

        best_fitness = fitness_values.max()
        avg_fitness = fitness_values.mean()
        diversity = population_diversity(population)

        convergence_log.append({
            "generation": gen,
            "best_fitness": round(best_fitness, 4),
            "avg_fitness": round(avg_fitness, 4),
            "diversity": round(diversity, 4),
        })

        if verbose and gen % max(1, generations // 10) == 0:
            print(f"[GA] Gen {gen:4d} | Best: {best_fitness:.2f} | "
                  f"Avg: {avg_fitness:.2f} | Diversity: {diversity:.3f}")

        # --- Elitism: individu terbaik otomatis lolos
        next_population = list(get_elites(population, fitness_values, p["elitism_rate"]))

        # --- Isi sisa populasi lewat seleksi + crossover + mutasi
        while len(next_population) < p["population_size"]:
            parent1 = tournament_selection(population, fitness_values, p["tournament_k"])
            parent2 = tournament_selection(population, fitness_values, p["tournament_k"])

            child1, child2 = crossover(parent1, parent2, p["crossover_rate"])
            child1 = mutate(child1, p["mutation_rate"], n_jobs)
            child2 = mutate(child2, p["mutation_rate"], n_jobs)

            next_population.append(child1)
            if len(next_population) < p["population_size"]:
                next_population.append(child2)

        population = np.array(next_population[:p["population_size"]])

    # --- Evaluasi akhir untuk ambil solusi terbaik dari generasi terakhir
    final_fitness = evaluate_population(population, fitness_matrix)
    best_idx = np.argmax(final_fitness)
    best_individual = population[best_idx]
    best_fitness = final_fitness[best_idx]

    if verbose:
        print(f"\n[GA] Selesai. Fitness terbaik akhir: {best_fitness:.2f}")

    return best_individual, best_fitness, convergence_log


# ============================================================
# UTILITAS: DIVERSITY POPULASI
# (dipakai nanti oleh mekanisme adaptive switching GA<->PSO<->SA)
# ============================================================
def population_diversity(population: np.ndarray) -> float:
    """
    Mengukur keberagaman populasi sbg rata-rata proporsi gen unik per
    kolom (posisi graduate), dinormalisasi 0-1.
    0 = seluruh individu identik ; 1 = seluruh individu sangat beragam.
    """
    n_pop, n_genes = population.shape
    unique_ratios = [
        len(np.unique(population[:, col])) / n_pop
        for col in range(n_genes)
    ]
    return float(np.mean(unique_ratios))


# ============================================================
# UTILITAS: KONVERSI SOLUSI -> DATAFRAME HASIL MATCHING (Table 3)
# ============================================================
def solution_to_dataframe(individual: np.ndarray, df_grad: pd.DataFrame,
                           df_job: pd.DataFrame, fitness_matrix: np.ndarray) -> pd.DataFrame:
    """
    Mengubah satu individu/solusi (array index job) menjadi DataFrame
    hasil matching yang mudah dibaca -- format akhir menyerupai Table 3.
    """
    rows = []
    for i, job_idx in enumerate(individual):
        grad_id = df_grad.iloc[i]["graduate_id"]
        job_row = df_job.iloc[job_idx]
        rows.append({
            "graduate_id": grad_id,
            "job_id": job_row["job_id"],
            "company_id": job_row["company_id"],
            "position_name": job_row["position_name"],
            "fitness_score": round(fitness_matrix[i, job_idx], 2),
        })
    return pd.DataFrame(rows)


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

    np.random.seed(42)

    best_individual, best_fitness, log = run_ga(df_grad, df_job, generations=500)

    df_result = solution_to_dataframe(best_individual, df_grad, df_job,
                                       precompute_fitness_matrix(df_grad, df_job))

    out_path = "../output/ga_result.csv"
    log_path = "../output/ga_convergence_log.csv"

    df_result.to_csv(out_path, index=False)
    pd.DataFrame(log).to_csv(log_path, index=False)

    print(f"\n[TEST] Hasil matching GA disimpan: {out_path}")
    print(f"[TEST] Log konvergensi disimpan  : {log_path}")
    print(f"\n[TEST] Rata-rata fitness solusi terbaik: {best_fitness:.2f}")
    print(df_result.head())