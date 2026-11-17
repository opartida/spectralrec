import csv
import glob
from pathlib import Path

import numpy as np

from spectralrec.lowrank import compute_singular_values


def main():

    experiments = {}
    with open("reports/experiments.csv", newline="") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            experiments[row["run_id"]] = {}
            experiments[row["run_id"]]["reg"] = row["reg"]

    dir = Path("results")

    rows = []
    print(
        f"| {'reg':>5} | {'σ₁':>10} | {'σ_j / σ₁ ≥ 0.01':>15} | {'σ_j / σ₁ ≥ 0.001':>16} |"
    )
    print(f"|{'-' * 7}|{'-' * 12}|{'-' * 17}|{'-' * 18}|")
    for filename in glob.iglob(f"{dir}/*als_k100*.npz"):
        run_id = filename.split("results/")[1].split("_")[0]
        with np.load(file=filename) as model_factors:
            A = model_factors["user_factors"]
            B = model_factors["movie_factors"]
            Z = A @ np.transpose(B)
        singular_values = compute_singular_values(matrix=Z)
        sigma1 = singular_values[0]
        reg = experiments[run_id]["reg"]

        effective_singular_values1 = np.sum(singular_values / sigma1 > 0.01)
        effective_singular_values2 = np.sum(singular_values / sigma1 > 0.001)

        rows.append(
            (float(reg), sigma1, effective_singular_values1, effective_singular_values2)
        )

    rows.sort()
    for reg, sigma1, count1, count2 in rows:
        print(f"| {reg:>5} | {sigma1:>10.4f} | {count1:>15} | {count2:>16} |")


if __name__ == "__main__":
    main()
