# README

This repository implements a **Conditional Neural Process (CNP)** for few-shot regression of a robot arm’s end-effector and object positions.

---

## Model Definition

- **Inputs (context):**  
  - `(t, h, e_y, e_z, o_y, o_z)` ⟶ time `t`, object height `h`, end-effector `(e_y,e_z)`, object `(o_y,o_z)`  
- **Query:**  
  - `(t, h)` at which to predict positions  
- **Outputs:**  
  - **Mean** and **StdDev** for `(e_y, e_z, o_y, o_z)`  
- **Architecture:**  
  1. **Encoder:** MLP mapping each context point `(d_x+d_y=6)` → 128 D, with ReLU, 3 hidden layers  
  2. **Aggregate:** Mean-pooling over context embeddings ⟶ 128 D summary  
  3. **Decoder:** MLP mapping summary + query `(128+2)` → `2*d_y=8` outputs (means & log-stds), softplus+`min_std` ⟶ positive σ  
  4. **Loss:** Negative log-likelihood under a Normal distribution

---

## Data Collection & Training

1. **Collect Trajectories**  
   - Use `Hw5Env` (MuJoCo) to generate 100 random Bézier-curve trajectories  
   - Record high-level states `(e_y, e_z, o_y, o_z, h)` at each of ~100 timesteps  
   - Build “trajectory” arrays of shape `(T, 6)` = `[t, e_y, e_z, o_y, o_z, h]`

2. **Training Loop**  
   - **Hyperparameters:**  
     - `hidden_size=128`, `num_hidden_layers=3`, `min_std=0.1`  
     - `lr=5e-5`, **epochs**=100 000, **batch_size**=32  
   - Each batch: sample 32 trajectories, for each randomly pick  
     - `n_context ∈ [1, max_context]`  
     - `n_target = 1`  
   - Optimize NLL loss over targets

---

## Evaluation & Test Results

- **100 Random Tests**  
  - For each: pick a new trajectory, sample `n_context ∈ [1, 10]` and 1 query  
  - Compute **MSE** separately for end-effector vs object  
  - Aggregate over 100 runs  

- **Results Directory:**  
[MSE Bar Plot](src/test_results/mse_bar.png)