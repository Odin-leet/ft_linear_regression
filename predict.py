#!/usr/bin/env python3
"""
predict.py - prompts the user for a mileage and prints the estimated price
using: estimatePrice(mileage) = theta0 + theta1 * mileage

If train.py has not been run yet (no theta.csv found), theta0 and theta1
default to 0, as required by the spec.
"""

import csv
import os
import sys

THETA_PATH = "theta.csv"


def load_thetas(path):
    """Return (theta0, theta1), defaulting to (0.0, 0.0) if no trained model exists yet."""
    if not os.path.exists(path):
        return 0.0, 0.0
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            row = next(reader)
            return float(row[0]), float(row[1])
    except (StopIteration, ValueError, IndexError):
        print(f"Warning: '{path}' is malformed, defaulting theta0=0, theta1=0.")
        return 0.0, 0.0


def estimate_price(theta0, theta1, mileage):
    return theta0 + theta1 * mileage


def prompt_mileage():
    raw = input("Enter a mileage (km): ").strip()
    try:
        mileage = float(raw)
    except ValueError:
        print("Error: please enter a valid number.")
        sys.exit(1)
    if mileage < 0:
        print("Error: mileage cannot be negative.")
        sys.exit(1)
    return mileage


def main():
    theta0, theta1 = load_thetas(THETA_PATH)
    if theta0 == 0.0 and theta1 == 0.0:
        print("(No trained model found yet — run train.py first for a real estimate.)")

    mileage = prompt_mileage()
    price = estimate_price(theta0, theta1, mileage)
    price = max(price, 0.0)  # a car's price can't sensibly be negative

    print(f"Estimated price for {mileage:.0f} km: {price:.2f}")


if __name__ == "__main__":
    main()
