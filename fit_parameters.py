import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

# Load experimental data
DATA_FILE = "reaction_data.csv"


def load_data(path=DATA_FILE):
    df = pd.read_csv(path)
    return df["New Cells"].to_numpy(), df["Rate (%)"].to_numpy()


# Vectorized simulation of the crystal reaction model


def simulate(params, steps, grid_size=80, seed=0):
    """Simulate reaction counts for given parameters.

    Parameters correspond to probabilities p0..p6 where p0 is the
    probability when a cell has six unreacted neighbours and p6 is for zero.
    """
    rng = np.random.default_rng(seed)
    p_class = params[::-1]  # p_class[0] -> p6
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    counts = []
    rates = []
    for _ in range(steps):
        # Count unreacted neighbours
        neighbours = np.zeros_like(grid, dtype=np.int8)
        neighbours[:-1, :, :] += ~grid[1:, :, :]
        neighbours[1:, :, :] += ~grid[:-1, :, :]
        neighbours[:, :-1, :] += ~grid[:, 1:, :]
        neighbours[:, 1:, :] += ~grid[:, :-1, :]
        neighbours[:, :, :-1] += ~grid[:, :, 1:]
        neighbours[:, :, 1:] += ~grid[:, :, :-1]
        free = ~grid
        probs = p_class[neighbours]
        rand = rng.random(grid.shape)
        new_react = free & (rand < probs)
        n_new = np.count_nonzero(new_react)
        grid |= new_react
        counts.append(n_new)
        rates.append(grid.mean() * 100)
        if len(rates) >= 5 and rates[-1] >= 99.5:
            break
    return np.array(counts), np.array(rates)


def objective(params, true_counts, true_rates, seed=0):
    pred_counts, pred_rates = simulate(params, len(true_counts), seed=seed)
    # Align lengths
    m = min(len(pred_counts), len(true_counts))
    pred_counts = pred_counts[:m]
    pred_rates = pred_rates[:m]
    tc = true_counts[:m]
    tr = true_rates[:m]
    # Mean squared error of counts and rates
    return np.mean((pred_counts - tc) ** 2) + np.mean((pred_rates - tr) ** 2)


def fit_parameters():
    counts, rates = load_data()
    bounds = [(0.0, 1.0)] * 7
    result = differential_evolution(
        objective,
        bounds,
        args=(counts, rates),
        strategy="best1bin",
        maxiter=100,
        polish=True,
        seed=0,
    )
    print("Fitted parameters (p0..p6):")
    for i, p in enumerate(result.x):
        print(f"p{i} = {p:.6f}")
    return result


if __name__ == "__main__":
    fit_parameters()
