#!/usr/bin/env python3
"""
train.py - trains a simple linear regression model (price = theta0 + theta1 * mileage)
using batch gradient descent, and saves theta0/theta1 to theta.csv for predict.py to use.

Usage:
    python3 train.py [path/to/data.csv]
"""

import sys
import csv
import argparse

DEFAULT_DATA_PATH = "data.csv"
THETA_PATH = "theta.csv"
PLOT_PATH = "training_plot.png"
LEARNING_RATE = 0.1
EPOCHS = 1000


def load_data(path):
    """Read a two-column CSV (mileage, price) into two lists of floats."""
    mileages, prices = [], []
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header row: km,price
            for row in reader:
                if not row:
                    continue
                mileages.append(float(row[0]))
                prices.append(float(row[1]))
    except FileNotFoundError:
        print(f"Error: could not find dataset file '{path}'.")
        sys.exit(1)
    except (ValueError, IndexError):
        print("Error: dataset is malformed. Expected two numeric columns: km,price")
        sys.exit(1)

    if len(mileages) < 2:
        print("Error: dataset needs at least 2 data points.")
        sys.exit(1)

    return mileages, prices


def normalize(values):
    """Min-max normalize a list of values to the [0, 1] range.

    Gradient descent on raw mileage values (tens of thousands) is unstable
    with a fixed learning rate: theta1's gradient would be huge compared to
    theta0's, so the two need very different step sizes. Normalizing first
    keeps both gradients on the same scale, so a single learning rate works
    and convergence is fast and stable.
    """
    v_min, v_max = min(values), max(values)
    span = v_max - v_min if v_max != v_min else 1.0
    normalized = [(v - v_min) / span for v in values]
    return normalized, v_min, v_max


def estimate_price(theta0, theta1, mileage):
    return theta0 + theta1 * mileage


def gradient_descent(
    mileages_norm, prices_norm, learning_rate, epochs, record_history=False
):
    theta0, theta1 = 0.0, 0.0
    m = len(mileages_norm)
    cost_history = [] if record_history else None

    for _ in range(epochs):
        errors = [
            estimate_price(theta0, theta1, mileages_norm[i]) - prices_norm[i]
            for i in range(m)
        ]
        grad0 = sum(errors) / m
        grad1 = sum(errors[i] * mileages_norm[i] for i in range(m)) / m

        # simultaneous update: both use the OLD theta0/theta1 via the
        # errors computed above, so neither update affects the other's gradient.
        theta0 -= learning_rate * grad0
        theta1 -= learning_rate * grad1

        if record_history:
            cost_history.append(sum(e**2 for e in errors) / m)

    return theta0, theta1, cost_history


def denormalize_thetas(theta0_n, theta1_n, km_min, km_max, price_min, price_max):
    """Convert thetas trained on normalized data back into thetas that work
    directly on raw mileage/price, so predict.py needs no normalization logic."""
    km_span = km_max - km_min if km_max != km_min else 1.0
    price_span = price_max - price_min if price_max != price_min else 1.0

    theta1 = theta1_n * price_span / km_span
    theta0 = theta0_n * price_span + price_min - theta1 * km_min
    return theta0, theta1


def compute_r2(mileages, prices, theta0, theta1):
    """R^2: fraction of variance in price explained by the model. 1.0 = perfect fit."""
    predictions = [estimate_price(theta0, theta1, x) for x in mileages]
    mean_price = sum(prices) / len(prices)
    ss_res = sum((prices[i] - predictions[i]) ** 2 for i in range(len(prices)))
    ss_tot = sum((p - mean_price) ** 2 for p in prices)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 1.0


def plot_results(mileages, prices, theta0, theta1, cost_history, path):
    """Save a two-panel figure: (1) data points with the fitted line,
    (2) cost decreasing over epochs. Requires matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skipping --plot. Install it with:")
        print("  pip install matplotlib --break-system-packages")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    ax1.scatter(mileages, prices, color="#2b6cb0", s=45, zorder=3, label="Data")
    x_line = [min(mileages), max(mileages)]
    y_line = [estimate_price(theta0, theta1, x) for x in x_line]
    ax1.plot(x_line, y_line, color="#e53e3e", linewidth=2, label="Fitted line")
    ax1.set_xlabel("Mileage (km)")
    ax1.set_ylabel("Price")
    ax1.set_title("Data and fitted regression line")
    ax1.legend()
    ax1.set_ylim(bottom=0)

    ax2.plot(
        range(1, len(cost_history) + 1), cost_history, color="#2b6cb0", linewidth=2
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cost (normalized scale)")
    ax2.set_title("Cost over training")
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved training plot to '{path}'.")


def save_thetas(theta0, theta1, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["theta0", "theta1"])
        writer.writerow([theta0, theta1])


def main():
    parser = argparse.ArgumentParser(
        description="Train a linear regression model (price vs mileage)."
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        default=DEFAULT_DATA_PATH,
        help="Path to the CSV dataset",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Save a plot of the fit and the cost curve"
    )
    args = parser.parse_args()

    mileages, prices = load_data(args.data_path)
    mileages_norm, km_min, km_max = normalize(mileages)
    prices_norm, price_min, price_max = normalize(prices)

    theta0_n, theta1_n, cost_history = gradient_descent(
        mileages_norm, prices_norm, LEARNING_RATE, EPOCHS, record_history=args.plot
    )
    theta0, theta1 = denormalize_thetas(
        theta0_n, theta1_n, km_min, km_max, price_min, price_max
    )

    save_thetas(theta0, theta1, THETA_PATH)

    r2 = compute_r2(mileages, prices, theta0, theta1)

    print("Training complete.")
    print(f"  theta0 = {theta0:.6f}")
    print(f"  theta1 = {theta1:.6f}")
    print(f"  model precision (R^2) = {r2:.4f}")
    print(f"Saved to '{THETA_PATH}'. Run predict.py to estimate a price.")

    if args.plot:
        plot_results(mileages, prices, theta0, theta1, cost_history, PLOT_PATH)


if __name__ == "__main__":
    main()
