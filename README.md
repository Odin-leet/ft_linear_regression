# ft_linear_regression

A simple linear regression implemented **from scratch** — no `sklearn`, no `numpy.polyfit`, no prebuilt ML libraries. The model learns to predict a car's price from its mileage, using gradient descent that we implement by hand.

This README explains not just *how to run it*, but *why every part of the algorithm exists* — the full reasoning chain from a basic straight-line equation to a trained model.

---

## Table of contents

- [What this project does](#what-this-project-does)
- [How to run it](#how-to-run-it)
- [The core idea](#the-core-idea)
- [The math, step by step](#the-math-step-by-step)
- [Why we normalize the data](#why-we-normalize-the-data)
- [The learning rate](#the-learning-rate)
- [Why 1000 epochs](#why-1000-epochs)
- [Project structure](#project-structure)
- [What could go wrong](#what-could-go-wrong)

---

## What this project does

Two programs:

- **`train.py`** — reads a dataset of (mileage, price) pairs, runs gradient descent to find the best-fit line, and saves the result.
- **`predict.py`** — asks for a mileage, and returns the estimated price using that saved line.

The model is a single straight line:

```
estimatePrice(mileage) = theta0 + theta1 * mileage
```

- `theta0` — the intercept (predicted price at 0 km)
- `theta1` — the slope (how much price drops per extra km)

Before training, both default to `0`.

---

## How to run it

```bash
# 1. Train the model (reads data.csv, writes theta.csv)
python3 train.py

# Optional: also save a plot of the fit and the training curve
python3 train.py --plot

# 2. Predict a price for a given mileage
python3 predict.py
> Enter a mileage (km): 100000
Estimated price for 100000 km: 6353.80
```

If you run `predict.py` **before** `train.py`, it still works — `theta0` and `theta1` default to `0`, exactly as the spec requires, and you'll just get a prediction of `0`.

---

## The core idea

Linear regression is about finding **one straight line** that best fits a scatter of data points.

If you plot mileage (x-axis) against price (y-axis) for a bunch of cars, you'll see a downward-sloping cloud of dots — more mileage, lower price. There are infinitely many possible lines you could draw through that cloud. Training is the process of searching through all of them and finding the one that sits closest to the data, overall.

Once that one line is found, predicting is trivial: pick any mileage, plug it into the line's formula, read off the price.

**Gradient descent** is the *method* used to search for that line. Instead of directly solving for the best `theta0`/`theta1` in one shot, it starts with a bad guess (`theta0 = theta1 = 0`) and repeatedly nudges both values in the direction that makes the line fit better — like a blindfolded hiker feeling which way is downhill and taking small steps toward the bottom of a valley.

---

## The math, step by step

### 1. The line equation

```
y = a*x + b
```
renamed for this project:
```
estimatePrice = theta1 * mileage + theta0
```

### 2. The error, for one data point

```
error_i = estimatePrice(mileage_i) - price_i
```

Positive → overpredicted. Negative → underpredicted.

### 3. The cost function (Mean Squared Error / MSE)

We need ONE number describing how bad the *whole* line is, not just one point:

```
cost = (1/m) * sum( error_i^2 )
```

- **`m`** is the number of data points in the dataset (24 in the provided `data.csv`).
- We **square** the error rather than just summing it or using absolute value, because:
  - Squaring prevents positive and negative errors from cancelling out.
  - `x^2` is smooth everywhere (no sharp corners), so a clean, well-defined slope exists at every point — including exactly at the minimum. `|x|` has an undefined slope right at zero.
  - The size of the derivative of `x^2` (which is `2x`) shrinks the closer you are to the minimum. This makes training naturally slow down and settle instead of endlessly overshooting.

Low cost = good line. High cost = bad line. Training = finding the `theta0`, `theta1` that minimize this cost.

### 4. The derivative (why it tells you "which way is downhill")

A derivative answers: *"if I nudge this variable slightly, does the output go up or down, and how fast?"* Taking the partial derivative of `cost` with respect to `theta0` and `theta1` separately gives you the **gradient** — the exact uphill direction in the (theta0, theta1) landscape.

Deriving `cost` (using the chain rule, since error is squared) gives:

```
∂cost/∂theta0 = (2/m) * sum( error_i )
∂cost/∂theta1 = (2/m) * sum( error_i * mileage_i )
```

The difference between the two comes from **sensitivity**: bumping `theta0` by 1 always changes `error_i` by exactly 1 (it's unmultiplied in the formula). Bumping `theta1` by 1 changes `error_i` by however big that car's mileage is (since `theta1` is multiplied by `mileage_i`). That's why the `theta1` formula has the extra `× mileage_i` term and the `theta0` formula doesn't.

### 5. Turning the derivative into a step — `tmpθ0` / `tmpθ1`

The project's own formulas (dropping the constant `2`, which just gets absorbed into whatever learning rate you pick):

```
tmpθ0 = learningRate * (1/m) * sum( estimatePrice(mileage_i) - price_i )
tmpθ1 = learningRate * (1/m) * sum( (estimatePrice(mileage_i) - price_i) * mileage_i )
```

`tmpθ0`/`tmpθ1` are **not** part of the model — they're the one-epoch correction amount, recalculated fresh every epoch and discarded right after use. Think of them as a sticky note: "adjust theta by this much," used once, then thrown away.

### 6. Applying the update

```
theta0 = theta0 - tmpθ0
theta1 = theta1 - tmpθ1
```

**Why subtract?** The derivative points toward increasing cost (uphill). Subtracting it is what automatically sends you downhill — no matter which side of the valley you're currently standing on:

- If `tmp` is positive (you're on the rising side), subtracting decreases theta → moves back down.
- If `tmp` is negative (you're on the falling side), subtracting a negative = adding → increases theta → also moves back down.

This self-correcting behavior is *why* gradient descent doesn't need a manual "check which direction to go" step — the sign of the derivative and the subtraction handle it automatically, every time.

**Critical — simultaneous update:** both `tmpθ0` and `tmpθ1` must be computed using the *same*, unmodified `theta0`/`theta1`. If you updated `theta0` first and then used the *new* `theta0` to compute `tmpθ1`, you'd be computing a derivative for a point on the landscape you're no longer standing at — mathematically invalid.

### 7. Repeat — the epoch loop

One step doesn't reach the bottom of the valley — it just gets a little closer. Steps 2–6 repeat for many epochs (this project uses 1000). Each epoch is mathematically guaranteed to reduce cost (or leave it unchanged), provided the learning rate isn't too large — this isn't a hope, it's a direct consequence of "subtract the derivative."

---

## Why we normalize the data

Raw mileage values (up to 240,000) are far larger than raw price values (thousands). Since `tmpθ1` involves `error * mileage`, this makes `tmpθ1` explode into enormous numbers, and gradient descent **diverges** — `theta0`/`theta1` blow up toward infinity/NaN instead of converging, almost immediately, regardless of learning rate.

**The fix — min-max normalization**, applied to both mileage and price before training:

```
value_scaled = (value - min) / (max - min)
```

This squeezes every value into the range `[0, 1]`, so both `theta0`'s and `theta1`'s updates end up on a similar, reasonable scale — letting one learning rate work well for both.

After training finishes, the resulting "scaled-world" thetas are converted back into "raw-world" thetas, so `predict.py` can work directly with normal mileage numbers:

```
theta1_raw = theta1_scaled * (price_max - price_min) / (mileage_max - mileage_min)
theta0_raw = theta0_scaled * (price_max - price_min) + price_min - theta1_raw * mileage_min
```

This keeps all the scaling complexity contained inside `train.py` — `predict.py` stays as simple as `theta0 + theta1 * mileage`, no conversion logic needed on that side.

---

## The learning rate

The learning rate is **not calculated — it's a value you choose** before training starts (a "hyperparameter"). It controls how big a step you take each epoch:

- **Too high** → you overshoot past the bottom of the valley, and `theta0`/`theta1` oscillate wildly or diverge to infinity/NaN.
- **Too low** → each step is tiny; training crawls and may not reach good values within your epoch budget.
- **Just right** → steady, smooth progress toward the minimum.

This project uses `0.1`, which works well *because the data is normalized first*. Without normalization, no reasonable learning rate avoids divergence on this dataset (we tested learning rates as small as `1e-9` and still saw divergence — only around `1e-10` did it survive, and by then it was moving so slowly it had barely learned anything after thousands of epochs).

---

## Why 1000 epochs

1000 isn't a mathematically special number — it's a practical default. What actually matters is **convergence**: the point where cost stops meaningfully decreasing, because you've reached (or nearly reached) the bottom of the cost valley, where the derivative itself is close to zero.

For this small dataset (24 rows), cost typically drops fast in the first ~100–200 epochs, then flattens out — by epoch 1000, further epochs would only produce microscopic changes. You can verify this yourself by running `train.py --plot` and checking whether the cost curve has flattened by the final epoch.

Note: convergence means "the best possible *straight line* fit has been found" — not "predictions are perfect." Real car prices depend on more than mileage alone (condition, brand, age, etc.), so some error remains even at the true minimum.

---

## Project structure

```
.
├── data.csv      # dataset: km,price pairs
├── train.py      # trains the model, saves theta0/theta1 to theta.csv
├── predict.py    # reads theta.csv, prompts for mileage, prints estimated price
└── theta.csv     # created by train.py — NOT committed with starting values, generated on first run
```

### `train.py`, function by function

| Function | Role |
|---|---|
| `load_data` | Reads `data.csv` into two lists: `mileages[]`, `prices[]` |
| `normalize` | Min-max scales a list to `[0, 1]`, returns the scaled list plus the original min/max |
| `estimate_price` | The model formula: `theta0 + theta1 * mileage` |
| `gradient_descent` | The training loop: computes errors, gradients (`tmpθ0`/`tmpθ1`), and updates `theta0`/`theta1`, once per epoch |
| `denormalize_thetas` | Converts scaled-space thetas back to raw-mileage-space thetas |
| `compute_r2` | Reports how well the final line fits the data (0 = useless, 1 = perfect) — a bonus sanity check, not required by the core algorithm |
| `plot_results` | Optional (`--plot`): saves a figure showing the data + fitted line, and the cost curve over training |
| `save_thetas` | Writes the final `theta0`, `theta1` to `theta.csv` |

### `predict.py`, function by function

| Function | Role |
|---|---|
| `load_thetas` | Reads `theta0`/`theta1` from `theta.csv`, defaulting to `(0, 0)` if the file doesn't exist yet |
| `estimate_price` | Same formula as in `train.py` — must match exactly |
| `prompt_mileage` | Asks the user for a mileage, validates it's a real, non-negative number |
| `main` | Loads thetas → prompts for mileage → computes and prints the estimated price |

---

## What could go wrong

- **Forgetting to normalize** → training diverges to infinity/NaN almost immediately on this dataset's scale.
- **Non-simultaneous updates** → using an already-updated `theta0` to compute `tmpθ1` corrupts the gradient calculation for that epoch.
- **Learning rate too high** → oscillation or divergence. Too low → training barely moves within the epoch budget.
- **Mismatched formulas between `train.py` and `predict.py`** → since the saved `theta0`/`theta1` are only valid for the exact formula shape they were trained under (`theta0 + theta1 * mileage`), predict.py must use the identical formula, in the identical order.
- **`predict.py` run before `train.py`** → not an error — by design, it falls back to `theta0 = theta1 = 0`, per the project spec.
