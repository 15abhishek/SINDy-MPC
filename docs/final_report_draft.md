# Nominal vs. Learned Dynamics for RRT-Guided Model Predictive Control in BeamNG

**Author:** Abhishek Joshi  
**Affiliation:** University of Washington — ME569 (Data-Driven Control) & ME548 (Linear Multivariable Control)  
**Date:** June 2026

---

## I. Abstract

Autonomous ground navigation in unstructured environments demands both accurate prediction of vehicle motion and real-time enforcement of safety constraints [1], [2]. First-principles models such as the kinematic bicycle provide interpretable structure for optimization-based control, yet they systematically under-represent simulator-specific effects and actuator nonlinearities [3], [4]. Conversely, deep learning approaches can fit rich behaviors from data but sacrifice interpretability and complicate integration with gradient-based model predictive control (MPC) [5], [6]. This project addresses that gap by discovering explicit vehicle dynamics with Sparse Identification of Nonlinear Dynamics (SINDy) [7], [8] from BeamNG.tech telemetry, exporting the learned equations into a JAX-native plant model, and embedding them in a two-layer navigation stack: Rapidly-exploring Random Trees (RRT) for global planning [9] and Sequential Quadratic Programming MPC (SQP-MPC) or Model Predictive Path Integral (MPPI) control [10], [11] for local tracking. Three engineering iterations proved essential: Savitzky–Golay smoothing replaced raw finite-difference derivatives that amplified simulator noise; a custom trigonometric feature library replaced default polynomial libraries that failed to recover rigid kinematic structure; and a coefficient-based JAX bridge replaced direct PySINDy evaluation so that `jax.jacobian` linearizations remain available for SQP-MPC. Closed-loop evaluation uses a *digital twin* workflow in which Python spawns physical obstacle cylinders in BeamNG from the same 2D obstacle disks used by RRT, enabling high-fidelity comparison of nominal bicycle, SINDy-learned, SQP-MPC, and MPPI controllers on identical reference paths. Results show that a kinematic SINDy library recovers bicycle-equivalent structure with coefficients near unity, achieves comparable open-loop prediction to the nominal model at cruise speeds, and supports successful goal reaching in live simulation—while highlighting sensitivity to derivative quality, library design, and the actuator mapping layer between MPC commands and simulator throttle/brake.

---

## II. Introduction

### A. Motivation and Background

Safe navigation in dynamic, partially structured environments remains a central challenge in field robotics and intelligent transportation [1], [2], [12]. Off-road and simulator-rich settings expose vehicles to terrain variation, unmodeled tire slip, and actuator latency that nominal analytical models struggle to capture [3], [13], [14]. Traditional approaches rely on carefully identified physical models whose parameters must be tuned per platform; when those assumptions break, tracking performance and constraint satisfaction degrade [4], [15].

The research community has responded with data-driven alternatives spanning deep reinforcement learning (DRL), imitation learning, and system identification [5], [16], [17]. Frameworks such as Model Predictive Adversarial Imitation Learning (MPAIL) unify reward inference with planning-based control and improve sample efficiency relative to pure policy learning [18]. Yet black-box neural policies remain difficult to certify, data-hungry, and fragile under distribution shift [6], [19]. Gaussian-process and subspace-identification methods offer complementary uncertainty-aware modeling [3], [15], while Koopman-based approaches linearize nonlinear dynamics in lifted observable spaces [20], [21]. Among interpretable alternatives, SINDy discovers sparse governing equations by regressing state derivatives against a library of candidate nonlinear functions [7], [8], [22]. Because SINDy yields *explicit algebraic* dynamics rather than opaque network weights, it aligns naturally with physics-informed learning and with optimization-based control that requires differentiable rollouts [22], [23].

At the control layer, MPC is the de facto standard for constrained trajectory tracking and obstacle avoidance [24]–[26]. Analytical SQP-MPC linearizes nonlinear dynamics along a nominal trajectory and solves a sequence of convex quadratic programs, providing stronger constraint handling than purely sampling-based methods in many regimes [27], [28]. MPPI draws control sequences from a stochastic distribution and weights them by trajectory cost, enabling aggressive maneuvering with complex cost landscapes [10], [11]. Recent surveys emphasize model improvement from data as the dominant integration path for learning-based MPC [5], [29], while SINDy-RL demonstrates that sparse dictionary models can match DRL performance with far fewer environment interactions [6], [30]. The present project follows a complementary route—*explicit SINDy dynamics coupled directly to analytical SQP-MPC*—rather than end-to-end reinforcement learning.

### B. Project Goals

Per the ME569 proposal, this work implements a two-layer architecture for an Ackermann-steered vehicle: (i) a high-level collision-free planner (RRT [9]); (ii) telemetry collection and SINDy/SINDYc identification [7], [8]; (iii) SQP-MPC with `jax.jacobian` linearizations; and (iv) comparative evaluation of nominal kinematic bicycle dynamics versus SINDy-learned dynamics on trajectory efficiency, constraint satisfaction, and computation time in offline and BeamNG closed-loop tests. A stretch goal—Learned Neural Network (LRN) dynamics—was deferred to stabilize the SINDy–MPC integration and BeamNG evaluation matrix.

### C. Project Scope and Course Delineation

This project is jointly submitted to ME569 and ME548. The following delineation matches syllabus requirements and reflects how responsibilities were partitioned in implementation.

**ME569 (Data-Driven Control) Scope:**

- Telemetry data collection in BeamNG.tech (`scripts/01_generate_data.py`, `sindympc/data/telemetry.py`)
- Noise mitigation via Savitzky–Golay filtering of state derivatives (`SINDyDerivative`, `sindympc/system_id/sindy_training.py`)
- Custom trigonometric feature library design (`BicycleFeatureLibrary`, `sindympc/system_id/sindy_features.py`)
- STLSQ sparse regression to recover governing equations and export to `data/sindy_model.npz` (`scripts/02_train_sindy.py`)

**ME548 (Linear Multivariable Control) Scope:**

- The `SINDyJAXModel` bridge for runtime plant evaluation (`sindympc/dynamics/sindy_model.py`)
- Analytical linearizations using `jax.jacobian` on RK4 discrete dynamics (`linearize_dynamics`, `sindympc/controllers/sqp_mpc.py`)
- Receding-horizon SQP-MPC formulation via CVXPY with obstacle and input constraints
- MPPI baseline comparison (`sindympc/controllers/mppi.py`)
- RRT global planner (`sindympc/planners/rrt.py`)

**Synergy:** SINDy produces explicit algebraic equations that can be exported as JAX coefficient tensors, enabling the ME548 MPC stack to linearize and optimize against learned dynamics without a black-box neural forward model [7], [22].

---

## III. Technical Background

### A. Sparse Identification of Nonlinear Dynamics (SINDy)

Consider continuous-time dynamics \(\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u})\) with state \(\mathbf{x} \in \mathbb{R}^n\) and control \(\mathbf{u} \in \mathbb{R}^m\). SINDy approximates \(\mathbf{f}\) as a sparse linear combination of columns in a feature library \(\boldsymbol{\Theta}(\mathbf{x}, \mathbf{u})\):

\[
\dot{\mathbf{x}} \approx \boldsymbol{\Theta}(\mathbf{x}, \mathbf{u}) \,\boldsymbol{\Xi},
\]

where \(\boldsymbol{\Xi}\) is identified by sparse regression—typically Sequential Thresholded Least Squares (STLSQ)—from trajectories and estimated derivatives [7]. SINDYc extends the formulation to systems with control inputs [8]. For MPC, the identified model must support Jacobian-based linearization of a discrete-time rollout map [22], [23].

### B. Kinematic Bicycle Model

The planar kinematic bicycle with state \([x, y, \theta, v]^\top\) and control \([a, \delta]^\top\) (longitudinal acceleration and front steering angle) is:

\[
\dot{x} = v\cos\theta,\quad \dot{y} = v\sin\theta,\quad \dot{\theta} = \frac{v\tan\delta}{L},\quad \dot{v} = a,
\]

where \(L\) is the wheelbase. This model serves as both the **nominal plant** and the **target structure** for the kinematic SINDy library [13], [28].

### C. SQP-MPC

Receding-horizon MPC solves, at each time step,

\[
\min_{\mathbf{u}_{0:H-1}} \sum_{k=0}^{H-1} \ell(\mathbf{x}_k, \mathbf{u}_k) + \ell_f(\mathbf{x}_H)
\quad \text{s.t.}\quad \mathbf{x}_{k+1} = f(\mathbf{x}_k, \mathbf{u}_k),\; \mathbf{g}(\mathbf{x}_k) \leq 0,
\]

then applies \(\mathbf{u}_0^\star\). SQP linearizes \(f\) about \((\bar{\mathbf{x}}, \bar{\mathbf{u}})\) to obtain \(\mathbf{x}_{k+1} \approx \mathbf{A}_k \mathbf{x}_k + \mathbf{B}_k \mathbf{u}_k + \mathbf{c}_k\), solves a convex QP (CVXPY), and iterates [24], [27]. Our implementation uses `jax.jacobian` on a JAX RK4 step for \((\mathbf{A}, \mathbf{B}, \mathbf{c})\) for both bicycle and SINDy plants.

### D. Model Predictive Path Integral (MPPI) Control

MPPI approximates optimal control by importance-weighted sampling of control sequences [10], [11]. At each step, \(K\) perturbed sequences are rolled forward; costs \(S_i\) yield weights \(w_i \propto \exp(-S_i / \lambda)\); and the applied control is the weighted average of sampled first controls. MPPI tolerates non-convex costs at the expense of hard constraint guarantees [10], [29]. Our MPPI shares the RK4 plant, reference path, and obstacle penalties with SQP-MPC for fair comparison.

---

## IV. Methods

### A. Data-Driven System Identification [ME569 Focus]

#### 1) Data Collection

Training data were collected in BeamNG.tech on GridMap v2 using an excitation script with varied throttle and steering at 100 Hz (`scripts/01_generate_data.py`). Logged signals include position, velocity, quaternion orientation, and actuator inputs, processed into state \([x, y, \theta, v]\) and controls. For SINDYc in kinematic mode, longitudinal acceleration \(a\) is estimated from smoothed velocity; in residual mode, raw throttle is retained.

#### 2) SINDy Training Pipeline

Training (`sindympc/system_id/sindy_training.py`) follows the SINDYc workflow [8]: subsample trajectories; differentiate states; build \(\boldsymbol{\Theta}\); fit with STLSQ (threshold 0.002 kinematic, 0.02 residual); audit sign consistency of \(\dot{x}, \dot{y}, \dot{\theta}\); export coefficients to `data/sindy_model.npz`.

#### 3) Engineering Iterations: Failures and Fixes

**Iteration 1 — Raw finite differences amplified simulator noise.** PySINDy's default finite-difference estimator on 100 Hz logs amplified high-frequency jitter in position and heading. STLSQ either zeroed all terms or selected spurious polynomial features. **Fix:** Savitzky–Golay smoothing differentiation (`SINDyDerivative(kind="savitzky_golay")`) with odd window length scaled to ~1 s (`SG_POLYORDER = 3`) [7], [22].

**Iteration 2 — Default polynomial libraries missed kinematic structure.** A generic cubic library on \([x, y, \theta, v, u_0, \delta]\) fit in-sample data but did not enforce trigonometric couplings (\(v\cos\theta\), \(v\sin\theta\), \(v\tan\delta/L\)). **Fix:** `BicycleFeatureLibrary` with **kinematic mode** (minimal basis \(\{v\cos\theta,\; v\sin\theta,\; v\tan(\delta)/L,\; a\}\)) and **residual mode** (kinematic block plus degree-2 polynomials and extra \(\{\sin\theta, \cos\theta, \tan\delta\}\) terms) [22].

**Iteration 3 — PySINDy NumPy models broke JAX autodiff.** PySINDy callables prevented end-to-end `jax.jacobian` on the RK4 map, forcing slow finite-difference linearization. **Fix:** Export \(\boldsymbol{\Xi}\) and library metadata to NPZ; reconstruct \(\dot{\mathbf{x}} = \boldsymbol{\Theta}\boldsymbol{\Xi}\) entirely in JAX at runtime (`sindympc/dynamics/sindy_model.py`), matching the bicycle module interface.

[Insert Figure: SINDy training pipeline — data flow from BeamNG logs through SG filter, library, STLSQ, to NPZ export]

### B. Optimization-Based Control [ME548 Focus]

#### 1) System Architecture

| Layer | Module | Role |
|-------|--------|------|
| Planning | `sindympc/planners/rrt.py` | 2D holonomic RRT with circular obstacles |
| Control | `sindympc/controllers/sqp_mpc.py`, `mppi.py` | SQP-MPC (CVXPY + JAX) and MPPI |
| Plants | `bicycle_model.py`, `sindy_model.py` | Nominal and learned JAX dynamics |
| Evaluation | `scripts/07_run_beamng_sim.py`, `08_benchmark_beamng.py` | Live BeamNG loop and benchmarks |

**State:** \(x, y\) [m], \(\theta\) [rad] from velocity direction when \(v\) exceeds a threshold, \(v\) [m/s]. **MPC controls:** \([a, \delta]\). BeamNG actuation maps desired speed and steering through a PID throttle/brake layer (`sindympc/beamng/helpers.py`).

#### 2) JAX Bridge and SQP-MPC

The SINDy–JAX bridge loads NPZ coefficients and evaluates \(\boldsymbol{\Theta}(\mathbf{x}, \mathbf{u})\) without PySINDy at runtime. `linearize_dynamics` computes Jacobians of the RK4 step for both plants. SQP-MPC selects `get_dynamics("bicycle" | "sindy")`, tracks a speed-scheduled reference along an RRT polyline (Chaikin-smoothed), and enforces box constraints on \(a, \delta\) plus disk obstacle constraints linearized each SQP iteration.

#### 3) MPPI Baseline

MPPI uses \(K=256\) samples, temperature \(\lambda\), and Gaussian perturbations on control sequences with softmax weighting. The same RK4 plant hook ensures controller comparisons isolate algorithm choice rather than integration differences [10], [11].

#### 4) BeamNG Digital Twin Evaluation

The closed-loop script (`scripts/07_run_beamng_sim.py`): (1) connects to BeamNG.tech; (2) plans with RRT and optionally reuses `rrt_path.json`; (3) spawns procedural obstacle cylinders at the same \((x, y, r)\) disks used in 2D planning; (4) runs at 10 Hz—poll state, solve MPC/MPPI, map \((a,\delta)\) to throttle/brake/steering, log JSONL telemetry. This *digital twin* isolates dynamics and controller effects while validating 2D plans in 3D physics [12], [13].

[Insert Figure: Two-layer architecture — RRT reference, SQP/MPPI loop, BeamNG actuator mapping]

---

## V. Results and Simulation

### A. Learned Model Structure

With kinematic library mode, SINDy recovers sparse coefficients aligned with the nominal bicycle: \(\dot{x} \approx c_x \cdot v\cos\theta\), \(\dot{y} \approx c_y \cdot v\sin\theta\), \(\dot{\theta} \approx c_\theta \cdot v\tan(\delta)/L\), \(\dot{v} \approx c_a \cdot a\), with \(c_x, c_y, c_\theta, c_a\) typically within a few percent of unity after STLSQ. Residual-mode models activate additional polynomial terms at medium speeds where tire effects depart from pure kinematics [3], [13].

[Insert Figure: SINDy coefficient table vs nominal bicycle]

[Insert Figure: Open-loop RK4 rollout overlay — logged vs bicycle vs SINDy on held-out telemetry]

### B. Offline Open-Loop Accuracy (Speed Bins)

One-step RK4 prediction RMSE was computed against logged telemetry, binned by speed (`scripts/08_benchmark_beamng.py --offline-only`):

| Speed bin | \(n\) | Bicycle RMSE \((x,y)\) [m] | SINDy RMSE \((x,y)\) [m] | Bicycle RMSE \(v\) | SINDy RMSE \(v\) |
|-----------|-------|------------------------------|--------------------------|--------------------|------------------|
| \(v < 2\) m/s | 287 | 0.017 | 0.019 | 0.034 | 0.154 |
| \(2 \leq v < 4\) m/s | 116 | 0.006 | 0.017 | 0.022 | 0.319 |
| \(v \geq 4\) m/s | 2596 | 0.030 | 0.031 | 0.020 | 0.123 |

At cruise speeds (\(v \geq 4\) m/s), SINDy and bicycle achieve **equivalent** position RMSE; SINDy velocity error remains higher due to throttle-to-acceleration mismatch in training data. Low-speed bins are dominated by heading noise, where both models struggle on \(\theta\).

[Insert Figure: Offline speed-bin RMSE — grouped bar chart, bicycle vs SINDy]

### C. BeamNG Closed-Loop Benchmarks

Live runs compare `sqp` vs `mppi` and `bicycle` vs `sindy` at target speeds 2.0, 3.5, and 5.0 m/s on a ~150 m route. Success is defined as final goal error \(< 5\) m.

**Representative controller × dynamics matrix:**

| Controller | Dynamics | Target [m/s] | Mean speed [m/s] | Mean CTE [m] | Goal err. [m] | Success |
|------------|----------|--------------|------------------|--------------|---------------|---------|
| SQP | SINDy | 2.0 | 1.86 | 3.75 | 0.49 | Yes |
| SQP | Bicycle | 2.0 | 1.90 | 3.68 | 0.49 | Yes |
| SQP | SINDy | 5.0 | 3.80 | 3.60 | 0.48 | Yes |
| SQP | Bicycle | 5.0 | 4.12 | 3.31 | 0.48 | Yes |
| MPPI | SINDy | 3.5 | 2.78 | 3.34 | 0.49 | Yes |
| MPPI | Bicycle | 5.0 | 4.11 | 3.45 | 0.49 | Yes |
| MPPI | SINDy | 2.0 | 1.35 | 3.45 | 5.23 | No |

**Fixed-path speed sweep (SINDy, shared RRT path, 1–10 m/s):** SQP + SINDy succeeded at 2–8 m/s (e.g., at 2.0 m/s: mean CTE **0.49 m**, mean speed **1.96 m/s**, goal error **0.50 m**). Failures at 1.0 m/s and above 8 m/s reflect actuator under-speed and aggressive reference scheduling rather than plant inaccuracy alone.

**Key observations:**

- **SINDy vs bicycle (SQP):** Comparable goal reaching and cross-track error; bicycle achieves slightly higher mean speed at elevated targets because its velocity channel is exact by construction.
- **SQP vs MPPI:** Both succeed at moderate speeds when actuator mapping is tuned; MPPI + SINDy at 2.0 m/s failed in one run (under-speed + terminal error), illustrating sensitivity to sampling noise and brake/throttle heuristics [10], [29].
- **Actuator layer dominates low-speed behavior:** Mean achieved speed frequently trails target, indicating the PID throttle/brake mapping—not the learned \(\dot{v}\) equation—is often the limiting factor.

[Insert Figure: Cross-track error comparison — SQP bicycle vs SINDy at 3.5 m/s]

[Insert Figure: Top-down BeamNG trajectories — RRT path, obstacles, executed trajectories]

[Insert Figure: Speed vs accuracy — target speed vs mean CTE and goal error (`speed_vs_accuracy.png`)]

[Insert Table: Mean / 95th-percentile SQP solve time [ms] — bicycle vs SINDy at 3.5 m/s]

---

## VI. Conclusions and Future Work

This project demonstrated a complete data-driven control stack for Ackermann-steered navigation: SINDy identification from BeamNG telemetry, explicit JAX export for gradient-based MPC, RRT global planning, and comparative evaluation of SQP-MPC and MPPI in a physics-rich digital twin.

The **ME569 contribution** is that successful identification for control depends as much on *derivative quality* and *library physics* as on sparse regression itself—Savitzky–Golay smoothing, trigonometric bicycle features, and the SINDy–JAX bridge were necessary to move from unusable NumPy models to real-time SQP linearizations. The **ME548 contribution** is that nominal and SINDy plants behave similarly in closed loop when the learned library embeds kinematic structure, but constraint satisfaction and speed tracking still hinge on MPC formulation details and the low-level actuator map.

**Lessons learned:**

1. Do not differentiate raw simulator telemetry without smoothing when identifying continuous-time models [7], [22].
2. Encode known geometry (sin/cos/tan structure) in the SINDy library before expecting STLSQ to discover it from polynomials alone [22], [23].
3. Separate training-time (PySINDy) from runtime (JAX) representations when MPC requires autodiff through rollouts [5], [29].
4. Benchmark in the same environment that generated training data, with shared obstacle semantics between planner and simulator [12], [13].

**Future work:** extend SINDYc across diverse maneuvers and terrain [8]; explore online R-SINDy [32]; add uncertainty-aware models (e.g., Gaussian processes [3]) for safety margins [29], [33]; implement the deferred LRN neural baseline [6], [27]; improve longitudinal actuator identification; and deploy dynamic-obstacle NMPC constraints [25], [26], [35].

---

## VII. Acknowledgments

The author thanks the ME569 and ME548 instructional staff at the University of Washington for feedback on the project proposal and literature review. [Insert specific instructor names, lab mates, BeamNG.tech access, and compute resources.]

---

## References

[1] C. Min et al., "Autonomous Driving in Unstructured Off-Road Environments: How Far Have We Come?" *J. Field Robot.*, 2025. doi:10.1002/rob.70154

[2] Z. Wu et al., "TADPO: Reinforcement Learning Goes Off-road," *arXiv:2603.05995*, 2026.

[3] S. Sathe et al., "Data Driven Vehicle Dynamics System Identification Using Gaussian Processes," SAE Tech. Paper 2024-01-2022, 2024. doi:10.4271/2024-01-2022

[4] [Anonymous], "Linear System Identification Versus Physical Modeling of Lateral–Longitudinal Vehicle Dynamics," IEEE. doi:10.1109/9098865

[5] L. Hewing, K. P. Wabersich, M. Menner, and M. N. Zeilinger, "Learning-Based Model Predictive Control: Toward Safe Learning in Control," *Annu. Rev. Control Robot. Auton. Syst.*, vol. 3, pp. 269–296, 2020. doi:10.1146/annurev-control-090419-075625

[6] N. Zolman et al., "SINDy-RL for interpretable and efficient model-based reinforcement learning," *Nat. Commun.*, vol. 16, p. 10714, 2025. doi:10.1038/s41467-025-65738-4

[7] S. L. Brunton, J. L. Proctor, and J. N. Kutz, "Discovering governing equations from data by sparse identification of nonlinear dynamical systems," *PNAS*, 2016. doi:10.1073/pnas.1517384113

[8] S. L. Brunton, J. L. Proctor, and J. N. Kutz, "Sparse Identification of Nonlinear Dynamics with Control (SINDYc)," *IFAC-PapersOnLine*, vol. 49, no. 18, pp. 710–715, 2016. doi:10.1016/j.ifacol.2016.10.249

[9] S. LaValle, "Rapidly-exploring random trees: A new tool for path planning," Research Report 9811, Iowa State Univ., 1998.

[10] G. Williams et al., "Aggressive driving with model predictive path integral control," in *Proc. IEEE ICRA*, 2016, pp. 1433–1440. doi:10.1109/ICRA.2016.7487277

[11] G. Williams et al., "Information theoretic MPC for model-based reinforcement learning," in *Proc. IEEE ICRA*, 2017, pp. 1714–1721. doi:10.1109/ICRA.2017.7989202

[12] C. Richter, J. Ware, and N. Roy, "High-speed autonomous navigation of unknown environments using learned probabilities of collision," in *Proc. IEEE ICRA*, 2014, pp. 6114–6121. doi:10.1109/ICRA.2014.6907760

[13] T. Han et al., "Dynamics Models in the Aggressive Off-Road Driving Regime," *arXiv:2405.16487*, 2024.

[14] M. Bajracharya et al., "Autonomous off-road navigation with end-to-end learning for the LAGR program," *J. Field Robot.*, vol. 26, no. 1, pp. 3–25, 2009. doi:10.1002/rob.20269

[15] S. Sathe et al., 2024; see also subspace identification in [4].

[16] L. Zhu et al., "A Multi-Objective Path Planning Framework in Off-Road Environments Based on Deep Reinforcement Learning," *IEEE Trans. Intell. Transp. Syst.*, vol. 26, no. 11, pp. 19163–19181, 2025. doi:10.1109/TITS.2025.3589586

[17] J. Sacks and B. Boots, "Learning to Optimize in Model Predictive Control," in *Proc. IEEE ICRA*, 2022, pp. 10549–10556. doi:10.1109/ICRA46639.2022.9812369

[18] T. Han et al., "Model Predictive Adversarial Imitation Learning for Planning from Observation," *arXiv:2507.21533*, 2025.

[19] N. Zolman et al., 2025; see [6].

[20] D. Bruder, C. D. Remy, and R. Vasudevan, "Nonlinear System Identification of Soft Robot Dynamics Using Koopman Operator Theory," in *Proc. IEEE ICRA*, 2019, pp. 6244–6250. doi:10.1109/ICRA.2019.8793766

[21] W. A. Manzoor, S. Rawashdeh, and A. Mohammadi, "Vehicular Applications of Koopman Operator Theory—A Survey," *IEEE Access*, vol. 11, pp. 25917–25931, 2023. doi:10.1109/ACCESS.2023.3257109

[22] M. Corbetta, "Application of sparse identification of nonlinear dynamics for physics-informed learning," in *Proc. IEEE Aerosp. Conf.*, 2020. doi:10.1109/AERO47225.2020.9172386

[23] M. Corbetta, 2020; physics-informed sparse regression, see [22].

[24] M. Sani et al., "Dynamic Obstacles Avoidance Using Nonlinear Model Predictive Control," in *Proc. IECON*, 2021. doi:10.1109/IECON48115.2021.9589658

[25] O. Y. Ismael et al., "Nonlinear Model Predictive Control-based Collision Avoidance for Mobile Robot," *J. Robot. Control*, 2024.

[26] B. Lindqvist et al., "Nonlinear MPC for Collision Avoidance and Control of UAVs With Dynamic Obstacles," *IEEE Robot. Autom. Lett.*, vol. 5, no. 4, pp. 6001–6008, 2020. doi:10.1109/LRA.2020.3010730

[27] D. Menegatti et al., "Tractable Data-Driven Model Predictive Control Using One-Step Neural Networks Predictors," *IEEE Trans. Autom. Sci. Eng.*, vol. 22, pp. 6740–6751, 2025. doi:10.1109/TASE.2024.3453668

[28] A. Nurbayeva et al., "Deep Imitation Learning of Nonlinear Model Predictive Control Laws for a Safe Physical Human–Robot Interaction," *IEEE Trans. Ind. Informat.*, vol. 19, no. 7, pp. 8384–8395, 2023. doi:10.1109/TII.2022.3217833

[29] L. Hewing et al., 2020; see [5].

[30] N. Zolman et al., 2025; see [6].

[31] O. Khatib, "Real-Time Obstacle Avoidance for Manipulators and Mobile Robots," *Int. J. Robot. Res.*, 1986. doi:10.1177/027836498600500106

[32] [Anonymous], "Recursive Sparse Identification of Nonlinear Dynamics for Online Equation Discovery," IEEE. doi:10.1109/10753612

[33] T. Koller et al., "Learning-Based Model Predictive Control for Safe Exploration," in *Proc. IEEE CDC*, 2018, pp. 6059–6066. doi:10.1109/CDC.2018.8619572

[34] D. Menegatti et al., 2025; see [27].

[35] Y. Du et al., "Nonlinear MPC of Quadrotors for Path Planning and Dynamic Obstacle Avoidance," in *Proc. IEEE CYBER*, 2024, pp. 485–490. doi:10.1109/CYBER63482.2024.10749189

---

*Document length: approximately 5–6 pages at 11 pt, 1 in margins, single column (excluding figures). Insert 4–6 figures to reach full submission length.*
