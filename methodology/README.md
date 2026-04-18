# Methodology

This approach follows Tanaka & Nagai's POD-PIML framework ([DOI: 10.1016/j.ijheatmasstransfer.2023.124336](https://doi.org/10.1016/j.ijheatmasstransfer.2023.124336)), adapted with two practical extensions: (i) a **hybrid supervised + physics loss** and (ii) a **device-level input interface** that maps directly to operational parameters.

---

## 1. Problem setting

Thermal design of a spacecraft requires solving a steady-state heat balance at every node of a thermal mathematical model (TMM). The baseline tool (Thermal Desktop + SINDA) solves this with a nonlinear finite-difference accurately, but it computationally expensive and time consuming. Robust design requires evaluating the response across orbit conditions × attitude × duty-cycle × device-power combinations, taking days of simulation time. A fast, physically-consistent surrogate model solves that issue.

At each of the `n = 264` nodes, steady state requires:

$$Q_{\text{in},i} \; - \; \sum_{j=1}^{n} C_{ij}\,(T_i - T_j) \; - \; \sigma \sum_{j=1}^{n} R_{ij}\,(T_i^4 - T_j^4) \; = \; 0$$

with

- $Q_{\text{in},i}$ — external heat load on node _i_ (solar, albedo, IR environment, and internal device dissipation)
- $C_{ij}$ — conductive conductance between nodes [W/K]
- $R_{ij}$ — radiative exchange factor [W/K⁴, with σ absorbed separately below]
- $\sigma = 5.67\times 10^{-8}$ W/m²/K⁴ — Stefan-Boltzmann constant

Both `C` and `R` nxn tensors extracted from QMAP node-connection files emitted by Thermal Desktop.

### 1.1 Node and Device Nomenclature

| Constant (`src/config.py`) | Value |
| -------------------------- | ----- |
| `NUM_NODES`                | 264   |
| `NUM_DEVICES`              | 7     |

The 7 devices and their node ranges:

| Device         | Nodes   | Baseline power |
| -------------- | ------- | -------------- |
| `Q_computer`   | 201–208 | 2 W            |
| `Q_powerboard` | 209–216 | 1 W            |
| `Q_avionics`   | 217–224 | 5 W            |
| `Q_battery`    | 225–240 | 5 W            |
| `Q_gnc`        | 241–248 | 2 W            |
| `Q_payload`    | 249–256 | 2 W            |
| `Q_radio`      | 257–264 | 1 W            |

A `mapping_matrix ∈ ℝ⁷ˣ²⁶⁴` distributes each device's total power uniformly across its constituent nodes. Environmental nodes use a pre-computed `Q_env` vector derived from the mean SINDA input across the training set.

---

## 2. Dataset

The data-generation pipeline extracts data from Sinda & randomly generates values for training.

| Dataset                | Size                                           | Purpose                                                      | Cost      |
| ---------------------- | ---------------------------------------------- | ------------------------------------------------------------ | --------- |
| **Prior** (SINDA runs) | 250 steady-state runs                          | Build POD basis; supervise the model on labeled (Q, T) pairs | Expensive |
| **Physics** (random Q) | 3,200 synthetic device-power vectors per epoch | Drive the physics loss without requiring solver output       | Free      |
| **Test** (SINDA)       | 500 steady-state runs                          | Final evaluation; never seen during training                 | Expensive |

Data generation is driven by a C# Visual Studio script that parameterizes Thermal Desktop sweeps. The pipeline then goes into the following data extraction and preprocessing scripts:

| Module                                                                                        | Role                                                        |
| --------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `training_node_connections_extraction.py`                                                     | Parses QMAP text files into node + connection CSVs          |
| `cij_rij_matrix_generation.py`                                                                | Builds `C_matrix`, `R_matrix` PyTorch tensors               |
| `Q_input_matrix_generation.py`                                                                | Assembles per-node heat-load matrix from SINDA output       |
| `T_prior_matrix_generation.py`                                                                | Assembles per-node temperature matrix from SINDA output     |
| `Q_random_generation.py`                                                                      | Samples synthetic device-power vectors for the physics loss |
| `test_node_data_extraction.py`, `Q_test_matrix_generation.py`, `T_sinda_matrix_generation.py` | Build the test data                                         |

Preprocessing: inputs are standardized with training-set mean and standard deviation (`X_mean`, `X_std`), cached in `models/tensors.pt` alongside the POD basis.

---

## 3. Dimensionality reduction via POD

We want to represent the full 264-node temperature field with far fewer numbers. POD does this by finding the recurring spatial patterns hidden inside our 250 SINDA runs.

Each column of the prior temperature matrix $\Theta_{\text{prior}} \in \mathbb{R}^{264 \times 250}$ is one SINDA run's temperature across all 264 nodes. We center the matrix by subtracting the mean temperature, then run SVD on it:

$$\Theta_{\text{prior}} - \bar{T} \; = \; U\,S\,V^{\top}$$

Variables:

- $\Theta_{\text{prior}}$: prior temperature matrix, shape 264 × 250 (rows are nodes, columns are SINDA runs), units K.
- $\bar{T}$: mean temperature vector across the 250 runs, shape 264 × 1, units K.
- $U$: left singular vectors, shape 264 × 264. Each column is one spatial temperature pattern (a "mode").
- $S$: diagonal matrix of singular values, shape 264 × 250. Entry $S_{kk}$ measures how much variance mode $k$ has.
- $V^{\top}$: right singular vectors, shape 250 × 250. Each row says how much of each mode appears in the matching SINDA run.

The diagonal of $S$ ranks the modes by importance. We keep the top 40 and discard the rest:

$$\Theta_{\text{prior}} - \bar{T} \; \approx \; U_{40}\,S_{40}\,V_{40}^{\top}, \qquad U_{40} \in \mathbb{R}^{264\times 40}$$

Variables:

- $U_{40}$: the first 40 columns of $U$, shape 264 × 40. These are the 40 most important spatial temperature patterns.
- $S_{40}$: the top 40 singular values, shape 40 × 40 diagonal.
- $V_{40}^{\top}$: the first 40 rows of $V^{\top}$, shape 40 × 250.

Once we have $U_{40}$, any temperature snapshot can be written as a short coefficient vector $\alpha \in \mathbb{R}^{40}$ instead of a 264-long vector:

$$T \; = \; \bar{T} + \alpha \cdot U_{40}^{\top}$$

Variables:

- $T$: reconstructed temperature vector for one snapshot, shape 264 × 1, units K.
- $\alpha$: POD coefficient vector, shape 40 × 1. This is what the neaural network will predict and output.
- $U_{40}^{\top}$: transpose of the POD basis, shape 40 × 264.

---

## 4. Network architecture

The neaural network takes 7 device powers in, predict 40 POD coefficients out. Temperatures get reconstructed afterwards by multiplying those coefficients back through $U_{40}$.

The surrogate is a small fully-connected MLP

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

- **SiLU activation.** Smooth gradients matter here because the physics loss takes derivatives through $T(\alpha)$ via the POD basis.
- **Output scaling by `diag(S_40)`** The raw network outputs land around order 1. Multiplying by the singular values puts each mode back at its real magnitude.
- **Last-layer near-zero initialization.** Otherwise the first forward pass multiplies random $\alpha$ by large singular values, producing temperatures and a physics loss that explodes in the first few epochs.
- **Device-level input (7 devices, not 264).** Inputs are the 7 device powers that an engineer sets. The `mapping_matrix` expands the 7-vector to the full 264-node $Q_{\text{in}}$ needed by the physics loss. This makes the surrogate directly usable by engineers iterating on operational configurations.

Temperatures are reconstructed as

$$\hat{T} \; = \; \bar{T} + \alpha \cdot U_{40}^{\top} \; = \; \bar{T} + \text{MLP}(\hat{x}) \cdot \text{diag}(S_{40}) \cdot U_{40}^{\top}$$

Variables:

- $\hat{T}$: predicted temperature vector, shape 264 × 1, units K.
- $\bar{T}$: mean temperature vector, shape 264 × 1, units K.
- $\alpha$: scaled POD coefficients produced by the network, shape 40 × 1.
- $\text{MLP}(\hat{x})$: raw network output before scaling, shape 40 × 1, order 1 in magnitude.
- $\hat{x}$: standardized 7-device input, shape 7 × 1.
- $\text{diag}(S_{40})$: diagonal matrix of the top 40 singular values, shape 40 × 40.
- $U_{40}^{\top}$: transpose of the POD basis, shape 40 × 264.

---

## 5. Loss function

The total loss blends two terms. A supervised term matches labeled SINDA runs, and a physics term forces predictions to obey the steady-state heat balance:

$$\mathcal{L} \ = \; \mathcal{L}_{\text{data}} \ + \; \lambda\,\mathcal{L}_{\text{phys}}, \qquad \lambda = 0.1$$

Variables:

- $\mathcal{L}$: total training loss, scalar.
- $\mathcal{L}_{\text{data}}$: supervised data loss, scalar.
- $\mathcal{L}_{\text{phys}}$: physics loss, scalar.
- $\lambda$: weighting factor. Controls how much the physics loss pulls on the network relative to the data loss.

### 5.1 Data loss (supervised on 250 SINDA runs)

For each training pair, we project the true SINDA temperature onto the POD basis to get the "target" coefficients the network should have produced. Then we train the network to match those targets.

$$\alpha^{\star}_i \; = \; (T_i - \bar{T}) \cdot U_{40}$$

Variables:

- $\alpha^{\star}_i$: target POD coefficients for training pair $i$, shape 40 × 1.
- $T_i$: true SINDA temperature vector for pair $i$, shape 264 × 1, units K.
- $\bar{T}$: mean temperature vector (same as section 3), shape 264 × 1, units K.
- $U_{40}$: POD basis, shape 264 × 40.

The data loss is mean squared error between the network's predicted coefficients and the targets:

$$\mathcal{L}_{\text{data}} \; = \; \frac{1}{N_{\text{sup}}} \sum_{i=1}^{N_{\text{sup}}} \bigl\| \hat{\alpha}_i - \alpha^{\star}_i \bigr\|^2_2$$

Variables:

- $\hat{\alpha}_i$: network's predicted POD coefficients for pair $i$, shape 40 × 1.
- $\alpha^{\star}_i$: target POD coefficients from above.
- $N_{\text{sup}}$: number of supervised training pairs per epoch, equal to 250.
- $\| \cdot \|^2_2$: squared Euclidean norm (sum of squared entries).

### 5.2 Physics loss (3,200 synthetic Q per epoch)

For each synthetic $Q$, we run the network to get a predicted temperature field, then plug it into the steady-state heat-balance equation. If the prediction is physically consistent, the residual is zero. If not, the network gets pushed to fix it.

For a batch of synthetic device-power vectors, we expand each one to a per-node $Q_{\text{in}}$ via the mapping matrix, pass it through the network to get $\hat{T}$, and compute the residual:

$$r(\hat{T}) \ = \ Q_{\text{in}} \-\ \hat{T} \cdot C \-\ \sigma\,\hat{T}^{\,4} \cdot R$$

Variables:

- $r(\hat{T})$: steady-state heat-balance residual per node, shape 264 × 1, units W.
- $Q_{\text{in}}$: per-node heat-load vector (external environment plus internal device dissipation), shape 264 × 1, units W.
- $\hat{T}$: predicted temperature vector, shape 264 × 1, units K.
- $C$: conductance matrix, shape 264 × 264, units W/K. The product $\hat{T} \cdot C$ gives the conductive heat flow out of each node in W.
- $\sigma$: Stefan-Boltzmann constant, $5.67 \times 10^{-8}$ W/m²/K⁴.
- $R$: radiative exchange factor matrix, shape 264 × 264. The product $\hat{T}^{\4} \cdot R$ (with $\sigma$ in front) gives the radiative heat flow out of each node in W.

We also add back the missing radiative exchange with deep space at $T_{\text{space}} = 3$ K. The TMM's $R$ matrix does not include a dedicated space node, so we patch that term in here.

The core physics loss is the MSE of the residual across the batch:

$$\mathcal{L}_1 \ = \ \text{MSE}\bigl(r(\hat{T})\bigr)$$

Variables:

- $\mathcal{L}_1$: mean squared heat-balance residual across the batch, scalar, units W².

On top of that, we add a hinge penalty for any predicted temperature below absolute zero (Eq. 17). It only activates when the network predicts something unphysical, and nudges it back up:

$$\mathcal{L}_2 \ = \ \begin{cases} \min(\hat{T})^2 & \text{if } \min(\hat{T}) < 0 \\ 0 & \text{otherwise} \end{cases}$$

Variables:

- $\mathcal{L}_2$: non-negativity penalty, scalar, units K².
- $\min(\hat{T})$: the smallest predicted temperature in the batch, units K.

The full physics loss is just the sum:

$$\mathcal{L}_{\text{phys}} \ = \; \mathcal{L}_1 + \mathcal{L}_2$$

---

## 6. Training

| Setting          | Value                       |
| ---------------- | --------------------------- |
| Optimizer        | Adam                        |
| Learning rate    | 1 × 10⁻⁴                    |
| Epochs           | 30 000                      |
| Supervised batch | 250 (full) per epoch        |
| Physics batch    | 3 200 random Q per epoch    |
| Physics weight λ | 0.1                         |
| Deep-space T     | 3.0 K                       |
| Activation       | SiLU                        |
| Hardware         | CUDA if available, else CPU |

**Best-model tracking.** During training we keep a running best checkpoint based on a combined validation metric, and save the winning weights to `models/spacecraft_therm_net.pth` at the end. The companion `models/tensors.pt` stores everything inference needs (`U_40`, `S_40`, `T_mean`, `X_mean`, `X_std`, the `mapping_matrix`, and `Q_env`), so loading the model is self-contained.

---

## 7. Evaluation

`pipeline/03_evaluate_model.py` reconstructs temperatures on the held-out 500-run SINDA test set and writes `data/processed/T_predict_matrix.csv` (264 nodes × 500 runs). The `notebooks/03_Model_Evaluation_and_Plots.ipynb` notebook produces the 9 publication figures.

### 7.1 Results

**Global error metrics** (132 000 predictions):

|           | Value   |
| --------- | ------- |
| MAE       | 1.11 K  |
| RMSE      | 2.24 K  |
| Median AE | 0.67 K  |
| Max AE    | 65.19 K |
| Mean Bias | −0.17 K |
| R²        | 0.9299  |
| MAPE      | 0.36%   |

**Per-run MAE** (averaged over all 264 nodes for each test run):

|                    | Run           | MAE     |
| ------------------ | ------------- | ------- |
| Best               | `run_224.out` | 0.05 K  |
| Median             | `run_192.out` | 0.72 K  |
| Worst              | `run_91.out`  | 13.35 K |
| Mean over 500 runs | —             | 1.11 K  |

**Physics consistency** on the test set:

|                    | Mean \|residual\| | Max \|residual\| |
| ------------------ | ----------------- | ---------------- |
| SINDA ground truth | 0.199 W           | 33.58 W          |
| PINN prediction    | 0.200 W           | 32.86 W          |

The PINN satisfies the steady-state heat balance to the same tolerance as SINDA itself — the solver's own convergence is the floor.

**Non-negativity**: 0 / 132 000 predictions below 0 K (min $\hat{T} = 225.22$ K).

<p align="center">
  <img src="../figures/median_run_detailed.png" width="780" alt="PINN prediction vs. SINDA ground truth across all 264 nodes for the median-difficulty test run.">
</p>

_Median run (`run_192`) — predicted vs. ground-truth profile over all 264 nodes, with per-node error in the lower panel._

---

## 8. Limitations

- **Steady-state only.** No transient dynamics; the paper's formulation already handles only the steady-state case, and extending to transient would require stepping the heat equation in time within the loss.
- **Held-out test set is SINDA, not flight data.** The surrogate's accuracy vs. reality is bounded by SINDA's own fidelity to the physical article.
- **Single spacecraft geometry.** `C`, `R`, and the mapping matrix are baked into the model; transfer to a new TMM requires retraining. A graph-neural-network formulation over the node connectivity graph is a natural generalization.
- **No formal ablation.** A clean comparison of (pure POD-PIML, pure POD-ANN, hybrid) on this exact 264-node TMM would be a useful follow-up — the hybrid's benefits are currently argued, not measured.
- **Worst-case failure mode.** `run_91` with 13.35 K MAE is ~12× the global mean. That run warrants individual diagnosis — likely an operational corner outside the prior data's convex hull, where the physics loss has to carry the prediction alone.

Directions under consideration: transient extension, a pure POD-PIML ablation, active learning to choose the next SINDA run adaptively, and a graph-conditioned architecture for transfer across TMMs.

---

## 9. References

1. **Tanaka, H., & Nagai, H.** (2023). _Thermal surrogate model for spacecraft systems using physics-informed machine learning with POD data reduction._ International Journal of Heat and Mass Transfer, 213, 124336. [DOI](https://doi.org/10.1016/j.ijheatmasstransfer.2023.124336). — Methodological foundation.
2. **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). _Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations._ Journal of Computational Physics, 378, 686–707. — Core PINN formulation.
3. **Berkooz, G., Holmes, P., & Lumley, J. L.** (1993). _The proper orthogonal decomposition in the analysis of turbulent flows._ Annual Review of Fluid Mechanics, 25, 539–575. — Foundational POD reference.
4. **Thermal Desktop / SINDA** — Cullimore & Ring Technologies (now C&R Tech). Commercial spacecraft thermal analysis software used to generate all training and test data.

---

**Author:** Martin Nguyen · Aerospace Engineering · San José State University
[GitHub](https://github.com/martinng06/SatelliteML) · [LinkedIn](https://www.linkedin.com/in/martinnguyen0/) · marngu06@gmail.com
