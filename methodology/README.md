# Methodology

## Problem setting

Thermal design of a small satellite requires solving a steady-state heat balance at every node of a finite-element model. The baseline tool (Thermal Desktop + SINDA) solves this with a sparse nonlinear finite-difference scheme — accurate, but on the order of minutes per configuration. Design-space exploration, uncertainty quantification, and trade studies need *thousands* of evaluations. A fast, physically-consistent surrogate closes that gap.

## Governing physics

At each of the N = 153 nodes, steady state requires:

```
Q_in,i  −  Σ_j C_ij (T_i − T_j)  −  Σ_j R_ij σ (T_i⁴ − T_j⁴)  =  0
```

- `Q_in,i` — external heat load on node *i* (solar, albedo, IR, internal dissipation)
- `C_ij` — conductance between nodes *i* and *j* [W/K]
- `R_ij` — radiative exchange factor [—]
- `σ` — Stefan-Boltzmann constant

Both `C` and `R` are sparse 153×153 matrices extracted from QMAP node connection files produced by Thermal Desktop.

## Dataset

- **Generator:** C# automation driver runs Thermal Desktop across a parameter sweep (orbit position, attitude, internal loads, optical properties).
- **Volume:** <FILL IN: N> successful runs.
- **Per run:** one heat-load vector `Q ∈ ℝ¹⁵³` and one steady-state temperature vector `T ∈ ℝ¹⁵³`.
- **Splits:** 70% train / 15% validation / 15% test.
- **Preprocessing:** `Q` normalized by training-set mean/std; stats stored alongside the model.

## Dimensionality reduction via POD

The training temperature matrix `T_train ∈ ℝ¹⁵³ˣᴺ` is factored by SVD:

```
T_train  =  U Σ Vᵀ
```

Truncating to **r = 40** modes captures >99% of the variance. The surrogate predicts mode coefficients `α ∈ ℝ⁴⁰`; temperatures reconstructed via:

```
T̂  =  U₄₀ · α
```

This keeps the network small (40 outputs instead of 153) and makes the physics loss cheap to evaluate — matrix products against a fixed basis.

## Network

`SpacecraftThermNet` (3-layer MLP):

```
Input  →  Linear(→150)  →  SiLU  →  Linear(→150)  →  SiLU  →  Linear(→40)
```

Input: normalized heat loads and any boundary-condition scalars. Output: 40 POD coefficients.

## Loss function

```
L  =  λ_data · L_data  +  λ_phys · L_phys  +  λ_nn · L_negative
```

- **`L_data`** — MSE between predicted `T̂` and ground-truth `T`.
- **`L_phys`** — MSE of the steady-state residual `r = Q_in − C·T̂ − R·σ·T̂⁴`. Enforces conservation of energy even in regions of input space where the data loss alone would smooth things over.
- **`L_negative`** — penalty for predictions below 0 K (unphysical). Small but nonzero in early training; approaches zero by convergence.

Weights `λ_*` tuned on the validation split. <FILL IN: final values if desired>

## Training

- Optimizer: <FILL IN>
- Epochs: <FILL IN>
- Batch size: <FILL IN>
- Hardware: <FILL IN>
- Convergence: validation MAE plateau

## Evaluation

Held-out 15% test runs. Metrics:
- **MAE** — mean absolute temperature error across all nodes and runs
- **Per-node MAE** — identifies nodes where the surrogate is weakest
- **Parity plot** — bias check
- **Physics residual distribution** — verifies `Q_in − Q_cond − Q_rad ≈ 0`

All figures live in [`../figures/`](../figures/) and update automatically.

## Limitations & next steps

- Trained on a single spacecraft geometry; transfer requires retraining (or a graph-network formulation).
- Steady-state only — no transient dynamics.
- Radiative exchange factors assumed geometry-fixed.
- <FILL IN: any other caveats>

Directions under consideration: transient extension, geometry-conditioned architecture (GNN over the node graph), active learning to reduce sweep cost.

## References

<FILL IN: a few key papers — original PINN work, POD surrogate examples, thermal-model references.>
