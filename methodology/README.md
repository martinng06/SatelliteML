# Methodology

A technical writeup of the Physics-Informed Neural Network (PINN) thermal surrogate for a 264-node spacecraft thermal mathematical model (TMM). The approach follows Tanaka & Nagai's POD-PIML framework ([DOI: 10.1016/j.ijheatmasstransfer.2023.124336](https://doi.org/10.1016/j.ijheatmasstransfer.2023.124336)), adapted with two practical extensions: (i) a **hybrid supervised + physics loss** and (ii) a **device-level input interface** that maps directly to operational parameters.

If you want the 60-second version, see the [main README](../README.md). This document is the full writeup.

---

## 1. Problem setting

Thermal design of a spacecraft requires solving a steady-state heat balance at every node of a thermal mathematical model (TMM). The baseline tool (Thermal Desktop + SINDA) solves this with a sparse nonlinear finite-difference scheme — accurate, but on the order of minutes per configuration on this 264-node model. Robust design requires evaluating the response across orbit conditions × attitude × duty-cycle × device-power combinations. The combinatorics are unreachable with the simulator alone. A fast, physically-consistent surrogate closes that gap.

At each of the `n = 264` nodes, steady state requires:

$$Q_{\text{in},i} \; - \; \sum_{j=1}^{n} C_{ij}\,(T_i - T_j) \; - \; \sigma \sum_{j=1}^{n} R_{ij}\,(T_i^4 - T_j^4) \; = \; 0$$

with

- $Q_{\text{in},i}$ — external heat load on node *i* (solar, albedo, IR environment, and internal device dissipation)
- $C_{ij}$ — conductive conductance between nodes [W/K]
- $R_{ij}$ — radiative exchange factor [W/K⁴, with σ absorbed separately below]
- $\sigma = 5.67\times 10^{-8}$ W/m²/K⁴ — Stefan-Boltzmann constant

Both `C` and `R` are sparse 264×264 tensors extracted from QMAP node-connection files emitted by Thermal Desktop.

### 1.1 Node and device vocabulary

| Constant (`src/config.py`) | Value |
|---|---|
| `NUM_NODES` | 264 |
| `NUM_DEVICES` | 7 |

The 7 devices and their node ranges:

| Device | Nodes | Baseline power |
|---|---|---|
| `Q_computer` | 201–208 | 2 W |
| `Q_powerboard` | 209–216 | 1 W |
| `Q_avionics` | 217–224 | 5 W |
| `Q_battery` | 225–240 | 5 W |
| `Q_gnc` | 241–248 | 2 W |
| `Q_payload` | 249–256 | 2 W |
| `Q_radio` | 257–264 | 1 W |

A `mapping_matrix ∈ ℝ⁷ˣ²⁶⁴` distributes each device's total power uniformly across its constituent nodes; environmental (non-device) nodes use a pre-computed `Q_env` vector derived from the mean SINDA input across the training set.

---

## 2. Dataset

The data-generation pipeline mirrors the paper's distinction between a *small, expensive* prior dataset and a *large, cheap* training dataset.

| Dataset | Size | Purpose | Cost |
|---|---|---|---|
| **Prior** (SINDA runs) | 250 steady-state runs | Build POD basis; supervise the model on labeled (Q, T) pairs | Expensive — runs the full simulator |
| **Physics** (random Q) | 3 200 synthetic device-power vectors per epoch | Drive the physics loss without requiring solver output | Free — no simulator calls |
| **Test** (held-out SINDA) | 500 steady-state runs | Final evaluation; never seen during training | Expensive but one-shot |

Data generation is driven by a C# Visual Studio automation project (`src/thermal dekstop script/`) that parameterizes Thermal Desktop sweeps. The private repo's `src/data/` modules then:

| Module | Role |
|---|---|
| `training_node_connections_extraction.py` | Parses QMAP text files into node + connection CSVs |
| `cij_rij_matrix_generation.py` | Builds sparse `C_matrix`, `R_matrix` PyTorch tensors |
| `Q_input_matrix_generation.py` | Assembles per-node heat-load matrix from SINDA output |
| `T_prior_matrix_generation.py` | Assembles per-node temperature matrix from SINDA output |
| `Q_random_generation.py` | Samples synthetic device-power vectors for the physics loss |
| `test_node_data_extraction.py`, `Q_test_matrix_generation.py`, `T_sinda_matrix_generation.py` | Build the held-out test set |

Preprocessing: inputs are standardized with training-set mean and standard deviation (`X_mean`, `X_std`), cached in `models/tensors.pt` alongside the POD basis.

---

## 3. Dimensionality reduction via POD

The prior temperature matrix $\Theta_{\text{prior}} \in \mathbb{R}^{264 \times 250}$ is centered and decomposed via SVD:

$$\Theta_{\text{prior}} - \bar{T} \; = \; U\,S\,V^{\top}$$

Truncating to $r = 40$ modes gives

$$\Theta_{\text{prior}} - \bar{T} \; \approx \; U_{40}\,S_{40}\,V_{40}^{\top}, \qquad U_{40} \in \mathbb{R}^{264\times 40}$$

and the temperature distribution for any snapshot can be expressed by a coefficient vector $\alpha \in \mathbb{R}^{40}$:

$$T \; = \; \bar{T} + \alpha \cdot U_{40}^{\top}$$

**Why 40 when 6 would suffice:** the singular-value spectrum of this 264-node TMM decays fast — 99.9% of the thermal variance is captured by the top 6 modes, and 99% by just 2. `r = 40` is intentionally over-specified so that small higher-order modes — which dominate localized gradients near heat-dissipating components — are preserved for the physics loss to act on. See `figures/pod_mode_analysis.png` for the spectrum.

`U_40`, `S_40`, and the centering mean $\bar{T}$ are cached in `models/tensors.pt` so training and inference use the same basis.

---

## 4. Network architecture

The surrogate is a small fully-connected MLP, `SpacecraftThermNet` (`src/models/pinn_model.py`):

```
Input (7 device powers, standardized)
   │
   ▼
Linear(7 → 150)  ──► SiLU
   │
   ▼
Linear(150 → 150) ──► SiLU
   │
   ▼
Linear(150 → 40)  ──► × S_40  ──►  α  (POD coefficients)
```

Design choices, and why:

- **SiLU activation** — matches the paper; smooth first derivative is desirable when the physics loss takes gradients through `T(α)` via the POD basis.
- **Output scaling by `diag(S_40)`** (paper Eq. 15) — the raw network outputs are O(1); multiplying by the singular values restores the correct per-mode magnitude so the network doesn't need to learn very large / very small numbers.
- **Last-layer near-zero initialization** — without this, initial $\alpha$ values multiplied by large singular values produce wild temperature fields and the physics loss explodes in the first few epochs. Zero-init keeps the start close to $\bar{T}$.
- **Device-level input** — inputs are the 7 device powers, not the 264-vector of per-node heat loads. The `mapping_matrix` expands the 7-vector to the full $Q_{\text{in}}$ for the physics loss. This makes the surrogate directly usable by engineers iterating on operational configurations.

Temperatures are reconstructed as

$$\hat{T} \; = \; \bar{T} + \alpha \cdot U_{40}^{\top} \; = \; \bar{T} + \text{MLP}(\hat{x}) \cdot \text{diag}(S_{40}) \cdot U_{40}^{\top}$$

---

## 5. Loss function

Total loss is a blend of a supervised data term (POD-ANN style) and the paper's physics term:

$$\mathcal{L} \; = \; \mathcal{L}_{\text{data}} \; + \; \lambda\,\mathcal{L}_{\text{phys}}, \qquad \lambda = 0.1$$

### 5.1 Data loss (supervised on 250 SINDA runs)

For each training pair `(Q_i, T_i)`, the target POD coefficients are obtained by projection:

$$\alpha^{\star}_i \; = \; (T_i - \bar{T}) \cdot U_{40}$$

and the data loss is MSE in coefficient space:

$$\mathcal{L}_{\text{data}} \; = \; \frac{1}{N_{\text{sup}}} \sum_{i=1}^{N_{\text{sup}}} \bigl\| \hat{\alpha}_i - \alpha^{\star}_i \bigr\|^2_2$$

### 5.2 Physics loss (3 200 synthetic Q per epoch)

Paper Eqs. 16–17, adapted with a deep-space boundary correction. For a batch of synthetic device-power vectors, expand to per-node $Q_{\text{in}}$ via the mapping matrix, pass through the network to get $\hat{T}$, and compute the steady-state heat-balance residual:

$$r(\hat{T}) \; = \; Q_{\text{in}} \;-\; \hat{T} \cdot C \;-\; \sigma\,\hat{T}^{\,4} \cdot R$$

with a correction term adding back the missing radiative exchange with deep space at $T_{\text{space}} = 3$ K (since the TMM's `R` matrix does not include a dedicated space node). The physics loss is

$$\mathcal{L}_1 \; = \; \text{MSE}\bigl(r(\hat{T})\bigr)$$

plus a hinge penalty for any predicted temperature below absolute zero (Eq. 17):

$$\mathcal{L}_2 \; = \; \begin{cases} \min(\hat{T})^2 & \text{if } \min(\hat{T}) < 0 \\ 0 & \text{otherwise} \end{cases}$$

$$\mathcal{L}_{\text{phys}} \; = \; \mathcal{L}_1 + \mathcal{L}_2$$

### 5.3 Why hybrid

Pure POD-PIML (paper) uses only $\mathcal{L}_{\text{phys}}$ and relies on the POD basis to regularize. Pure POD-ANN uses only $\mathcal{L}_{\text{data}}$ and requires many solver-generated labels. The hybrid used here benefits from both:

| Regime | What the loss gives you |
|---|---|
| Near training data | `L_data` locks the prediction onto known solutions |
| Far from training data | `L_phys` keeps the prediction physically consistent via the heat-balance residual |
| Radiation-dominated regions | `L_phys` enforces the σT⁴ term that a pure data loss can't "see" under distribution shift |
| Non-physical outputs | `L₂` clamps negative temperatures to zero |

---

## 6. Training recipe

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1 × 10⁻⁴ |
| Epochs | 30 000 |
| Supervised batch | 250 (full) per epoch |
| Physics batch | 3 200 random Q per epoch |
| Physics weight λ | 0.1 |
| Deep-space T | 3.0 K |
| Activation | SiLU |
| Hardware | CUDA if available, else CPU |

**Best-model tracking.** Training keeps a running best checkpoint based on a combined validation metric and writes the winning state to `models/spacecraft_therm_net.pth` at the end. The companion `models/tensors.pt` caches `U_40`, `S_40`, `T_mean`, `X_mean`, `X_std`, the `mapping_matrix`, and `Q_env` so inference is self-contained.

---

## 7. Evaluation

`pipeline/03_evaluate_model.py` reconstructs temperatures on the held-out 500-run SINDA test set and writes `data/processed/T_predict_matrix.csv` (264 nodes × 500 runs). The `notebooks/03_Model_Evaluation_and_Plots.ipynb` notebook produces the 9 publication figures.

### 7.1 Results

**Global error metrics** (132 000 predictions):

| | Value |
|---|---|
| MAE | 1.11 K |
| RMSE | 2.24 K |
| Median AE | 0.67 K |
| Max AE | 65.19 K |
| Mean Bias | −0.17 K |
| R² | 0.9299 |
| MAPE | 0.36% |

**Per-run MAE** (averaged over all 264 nodes for each test run):

| | Run | MAE |
|---|---|---|
| Best | `run_224.out` | 0.05 K |
| Median | `run_192.out` | 0.72 K |
| Worst | `run_91.out` | 13.35 K |
| Mean over 500 runs | — | 1.11 K |

**Physics consistency** on the test set:

| | Mean \|residual\| | Max \|residual\| |
|---|---|---|
| SINDA ground truth | 0.199 W | 33.58 W |
| PINN prediction | 0.200 W | 32.86 W |

The PINN satisfies the steady-state heat balance to the same tolerance as SINDA itself — the solver's own convergence is the floor.

**Non-negativity**: 0 / 132 000 predictions below 0 K (min $\hat{T} = 225.22$ K).

### 7.2 Figure-by-figure

Each figure lives in [`../figures/`](../figures/). One-line interpretations below.

<p align="center">
  <img src="../figures/median_run_detailed.png" width="780" alt="PINN prediction vs. SINDA ground truth across all 264 nodes for the median-difficulty test run.">
</p>

*Median run (`run_192`) — predicted vs. ground-truth profile over all 264 nodes, with per-node error in the lower panel.*

| Figure | What it shows |
|---|---|
| `parity_plot.png` | Hexbin density of predicted vs. SINDA temperature. Tight diagonal = low bias across the range. |
| `per_run_error_distribution.png` | Histogram + violin of per-run MAE across 500 test runs. Identifies whether errors concentrate in a few difficult cases. |
| `per_node_mae_sorted.png` | Per-node MAE sorted; top 20 worst nodes highlighted. Typically high-gradient regions near heat sources. |
| `error_heatmap.png` | (node, run) signed-error heatmap for the top 30 worst nodes — reveals systematic spatial / operational patterns. |
| `temperature_profiles_comparison.png` | Side-by-side profiles for best, median, worst test runs. |
| `median_run_detailed.png` | Detailed view of the median run with node-wise error bars (shown above). |
| `physics_residuals.png` | Distribution of the heat-balance residual for SINDA vs. PINN predictions. |
| `negative_temperature_distribution.png` | Histogram of sub-zero predictions (empty for this run — none occurred). |
| `pod_mode_analysis.png` | Singular-value spectrum and cumulative variance captured by the POD basis. |

---

## 8. Comparison to the paper

The paper evaluates three variants on 100-node (Model A) and 1464-node (Model B) TMMs: POD-PIML (physics loss only), POD-ANN (labeled data loss only), and POD-GPR (Gaussian process regression in coefficient space). This project's surrogate is a fourth point in that design space:

| | POD-PIML (paper) | POD-ANN (paper) | POD-GPR (paper) | **This project** |
|---|---|---|---|---|
| Data usage | Cheap random Q only | Labeled (Q, T) pairs | Labeled (Q, T) pairs | **Both** |
| Loss | $\mathcal{L}_{\text{phys}}$ | MSE on α | GP log-likelihood | $\mathcal{L}_{\text{data}} + 0.1\,\mathcal{L}_{\text{phys}}$ |
| Prior dataset size | Small (builds POD basis) | Same labels drive training | Same labels drive training | 250 SINDA runs |
| Training dataset size | Large, free | Same as prior (expensive) | Same as prior (expensive) | 250 + 3 200 synthetic per epoch |
| Training cost vs. POD-ANN | 26–81% lower | Baseline | Comparable | Intermediate |
| Input interface | Per-node heat-load vector | Per-node heat-load vector | Per-node heat-load vector | **7-device power vector** |
| Deep-space boundary | Implicit | N/A | N/A | **Explicit `T_space = 3 K` correction** |

Two practical extensions over the paper:

1. **Hybrid loss.** Combining the supervised and physics terms lets the data loss anchor coefficients where labels exist, while the physics loss generalizes behavior under input distributions the SINDA runs never covered. For a small 250-run prior, pure POD-PIML risks slow convergence and pure POD-ANN risks distributional brittleness; the blend hedges both.
2. **Device-level inputs.** Accepting 7 device powers (instead of 264 per-node heat loads) matches how thermal engineers actually parameterize the design. The expansion to the full per-node $Q_{\text{in}}$ happens inside the physics loss via the `mapping_matrix`, so the network stays small and the interface stays operational.

---

## 9. Limitations & next steps

- **Steady-state only.** No transient dynamics; the paper's formulation already handles only the steady-state case, and extending to transient would require stepping the heat equation in time within the loss.
- **Held-out test set is SINDA, not flight data.** The surrogate's accuracy vs. reality is bounded by SINDA's own fidelity to the physical article.
- **Single spacecraft geometry.** `C`, `R`, and the mapping matrix are baked into the model; transfer to a new TMM requires retraining. A graph-neural-network formulation over the node connectivity graph is a natural generalization.
- **No formal ablation.** A clean comparison of (pure POD-PIML, pure POD-ANN, hybrid) on this exact 264-node TMM would be a useful follow-up — the hybrid's benefits are currently argued, not measured.
- **Worst-case failure mode.** `run_91` with 13.35 K MAE is ~12× the global mean. That run warrants individual diagnosis — likely an operational corner outside the prior data's convex hull, where the physics loss has to carry the prediction alone.

Directions under consideration: transient extension, a pure POD-PIML ablation, active learning to choose the next SINDA run adaptively, and a graph-conditioned architecture for transfer across TMMs.

---

## 10. References

1. **Tanaka, H., & Nagai, H.** (2023). *Thermal surrogate model for spacecraft systems using physics-informed machine learning with POD data reduction.* International Journal of Heat and Mass Transfer, 213, 124336. [DOI](https://doi.org/10.1016/j.ijheatmasstransfer.2023.124336). — Methodological foundation.
2. **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686–707. — Core PINN formulation.
3. **Berkooz, G., Holmes, P., & Lumley, J. L.** (1993). *The proper orthogonal decomposition in the analysis of turbulent flows.* Annual Review of Fluid Mechanics, 25, 539–575. — Foundational POD reference.
4. **Thermal Desktop / SINDA** — Cullimore & Ring Technologies (now C&R Tech). Commercial spacecraft thermal analysis software used to generate all training and test data.

---

**Author:** Martin Nguyen · Aerospace Engineering, Physics minor · San José State University (Class of 2028)
[GitHub](https://github.com/martinng06/SatelliteML) · [LinkedIn](https://www.linkedin.com/in/martinnguyen0/) · marngu06@gmail.com
