# SINDy-MPC
### *Project Title:* Nominal vs. Learned Dynamics for RRT-Guided MPC

**A dual-course project bridging System Identification (ME569) and Optimal Control (ME548).**

This repository contains a JAX-accelerated control architecture that pairs the **Sparse Identification of Nonlinear Dynamics (SINDy)** algorithm with an optimization-based **Model Predictive Controller (SQP-MPC)** for high-speed dynamic obstacle avoidance. 

## 📖 Project Overview
Safe autonomous navigation in dynamic environments requires strict, real-time constraint enforcement. While black-box neural networks struggle with safety verification, analytical MPC offers mathematical guarantees—provided it has an accurate dynamic model. 

This project extracts an explicit, algebraic vehicle dynamics model from empirical data using PySINDy (ME569), and embeds those learned equations directly into a continuous-time, JAX-linearized SQP-MPC solver (ME548).

### 🛠 Architecture
1. **Simulation Environment:** A custom Python/JAX 2D environment with kinematic bicycle dynamics and moving obstacles.
2. **System ID (ME569):** PySINDy extracts explicit non-linear governing equations from randomly generated trajectory data.
3. **High-Level Planner (ME548):** RRT/APF generates an initial, collision-free reference trajectory.
4. **Low-Level Controller (ME548):** A JAX-accelerated Sequential Quadratic Programming (SQP) MPC tracks the path while enforcing hard obstacle constraints via `jax.jacobian` linearization.

## 📂 Repository Structure
```text
SINDyMPC/
├── docs/                       # LaTeX proposals, lit reviews, and final reports
├── data/                       # Simulated trajectory data for SINDy training
├── sindympc/                   # Main library
│   ├── dynamics/               # Nominal and SINDy-learned plant models (JAX)
│   ├── system_id/              # PySINDy training scripts (ME569)
│   ├── planners/               # RRT and APF algorithms
│   ├── controllers/            # SQP-MPC and MPPI implementations
│   └── utils/                  # Math and visualization helpers
├── scripts/                    # Executable scripts to run pipelines
│   ├── 01_generate_data.py     
│   ├── 02_train_sindy.py       
│   └── 03_run_mpc_sim.py       
├── requirements.txt            
└── README.md
