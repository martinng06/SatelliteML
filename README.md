# Physics Informed Hybrid Surrogate Model for Spacecraft Thermal Systems Application

A **Physics Informed Hybrid Surrogate Model** that predicts the steady-state temperature of all 264 nodes of a spacecraft thermal model from 7 device power inputs, trained on a hybrid of supervised SINDA simulation data and physics-informed heat-balance loss function.

**Martin Nguyen** — Aerospace Engineering, Physics Minor — San José State University  
[GitHub](https://github.com/martinng06/SatelliteML) · [LinkedIn](https://www.linkedin.com/in/martinnguyen0/) · marngu06@gmail.com

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Thermal Desktop](https://img.shields.io/badge/Thermal%20Desktop-SINDA-005F9E)
![Physics-Informed](https://img.shields.io/badge/Physics--Informed-ML-6a1b9a)

---

## Overview

Spacecraft thermal analysis is slow. A single steady-state run in Thermal Desktop / SINDA is computationally expensive and the combinatorics of attitude × orbit × duty-cycle × device-power make the full design space effectively unreachable with the thermal mathematical model simulation alone.

This project builds a **Physics-Informed Neural Network (PINN) surrogate** that collapses that cost to milliseconds. The pipeline ingests 250 steady-state SINDA runs, compresses the 264-node temperature response into 40 POD (proper orthogonal decomposition) modes via SVD, and trains a small neural network to predict the POD coefficients from device-level power inputs. A physics loss is blended with the supervised data loss so the network learns solutions obey the fundamental conduction and radiation laws of physics.

The approach follows Tanaka & Nagai's POD-PIML methodology (_International Journal of Heat and Mass Transfer_ 213, 2023), extended with a **hybrid supervised + physics loss** and a **device-level input interface** that maps directly onto operational parameters engineers vary.

---

## Methodology

1. **Generate data.** An **automated C# script** sweeps Thermal Desktop simulations across spacecraft operating points, producing 250 steady-state SINDA solutions for the "prior" dataset and 500 independent runs for the held-out test set.
2. **Compress.** SVD on the 264 × 250 training temperature matrix; keep the top 40 modes (>99.9% variance).
3. **Learn.** A 3-layer Neural Network (7 → 150 → 150 → 40, SiLU activations) maps standardized device powers to POD coefficients α. Temperatures reconstructed as `T = α · U₄₀ᵀ + T_mean`.
4. **Enforce physics.** Total loss `L = L_data + 0.1 · L_phys` where `L_phys` penalizes the residual of `Q_in − C·T − σ·R·T⁴` with a deep-space boundary correction at 3 K, plus a hinge penalty for any predicted T < 0 K.
5. **Evaluate.** 500-run test set. Parity, per-node MAE, per-run MAE, physics residuals, POD spectrum, non-negativity.

Full technical writeup: **[methodology/README.md](methodology/README.md)**.

---

## Results

<p align="center">
  <img src="figures/median_run_detailed.png" width="780" alt="Predicted vs. SINDA temperatures across all 264 nodes for the median-difficulty held-out test run, with per-node error residuals in the lower panel.">
</p>

_A representative test case from the 500-run held-out set. Top: predicted (dashed) vs. SINDA ground truth (solid) for all 264 nodes. Bottom: per-node error, bounded within ±2 K for the majority of nodes._

Held-out test set: **500 steady-state runs × 264 nodes = 132,000 predictions**.

| Metric                | Value      |
| --------------------- | ---------- |
| Mean Absolute Error   | **1.11 K** |
| Median Absolute Error | 0.67 K     |
| RMSE                  | 2.24 K     |
| Max Absolute Error    | 65.19 K    |
| Mean Bias             | −0.17 K    |
| R²                    | 0.9299     |
| MAPE                  | 0.36%      |

Per-run accuracy:

| Run class                   | MAE    |
| --------------------------- | ------ |
| Best test run (`run_224`)   | 0.05 K |
| Median test run (`run_192`) | 0.72 K |

Physics consistency on the test set:

|                     | Mean \|residual\| | Max \|residual\| |
| ------------------- | ----------------- | ---------------- |
| SINDA ground truth  | 0.199 W           | 33.58 W          |
| **PINN prediction** | **0.200 W**       | **32.86 W**      |

Non-negativity: **0 / 132 000** predictions below 0 K.

---

## Highlights

- **264-node thermal mesh → 40 POD modes** (99.9% variance captured by just 6 modes)
- **Hybrid loss** = Mean Squared Error on POD coefficients (250 SINDA pairs) + steady-state heat-balance residual on 3,200 synthetic device-power samples per epoch
- **7-device input interface** computer, powerboard, avionics, battery, GNC, payload, radio
- **Physics consistency verified**: PINN mean residual (0.20 W) matches SINDA results (0.20 W) on the test set
- **Zero non-physical predictions**: 0 / 132,000 predicted temperatures below 0 K
- **End-to-end pipeline**: C# Thermal Desktop automated data generation sweep → Python data extraction → SVD → PyTorch training → CSV prediction export
- **Post Processing** figures covering parity, per-node / per-run error, physics residuals, POD spectrum, and non-negativity

---

## Repository map

```
satellite-thermal-ml/
├── methodology/          Technical deep-dive: physics, POD, network, loss, training
├── figures/              Evaluation figures (auto-synced from private repo)
└── README.md
```

---

## Tech stack

- **Simulation:** Thermal Desktop + SINDA, driven by a C# script that sweeps through different simulation varaibles to generate dataset
- **ML:** PyTorch, NumPy, SciPy
- **Data pipeline:** Python modules for QMAP parsing, conductance-matrix assembly (C, R ∈ ℝ²⁶⁴ˣ²⁶⁴), and SVD
- **Analysis:** Matplotlib, Seaborn, Pandas, Jupyter
- **Training:** Adam optimizer, lr 1e-4, 30 000 epochs

---

## Contact

**Martin Nguyen**
Aerospace Engineering · San José State University
[github.com/martinng06/SatelliteML](https://github.com/martinng06/SatelliteML) · [linkedin.com/in/martinnguyen0](https://www.linkedin.com/in/martinnguyen0/) · marngu06@gmail.com
