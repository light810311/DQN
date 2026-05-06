
# 🧠 Deep Q-Network (DQN) — Homework 3

> **Course**: Reinforcement Learning  
> **Environment**: Gridworld  
> **Date**: 2026-05-05  
> **Frameworks**: PyTorch & TensorFlow/Keras

---


## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [HW3-1: DQN Evolution Report](#hw3-1-dqn-evolution-report)
4. [HW3-2: RL Comprehensive Analysis](#hw3-2-rl-comprehensive-analysis)
5. [HW3-3: PyTorch vs. Keras Comparison](#hw3-3-pytorch-vs-keras-comparison)
6. [HW3-3 Bonus: DQN Soft Update Analysis](#hw3-3-bonus-dqn-soft-update-analysis)
7. [Key Results Summary](#key-results-summary)
8. [References](#references)

---

## Project Overview

This repository documents **Homework 3** of the Reinforcement Learning course, exploring the evolution and variants of **Deep Q-Network (DQN)** algorithms applied to the **Gridworld** environment. Four reports cover:

- The step-by-step evolution of DQN (Naïve → Replay Buffer → Target Network)
- Comprehensive comparison of DQN, Double DQN, and Dueling DQN
- A cross-framework implementation comparison (PyTorch vs. Keras/TensorFlow)
- A bonus analysis of Soft Update vs. Hard Update for the target network

---

## Repository Structure

```
RL-HW3/
├── DQN.ipynb                          # PyTorch DQN (Listing 3.3, 3.5, 3.7)
├── DQN_keras.ipynb                    # Keras/TensorFlow DQN reimplementation
├── Double DQN.ipynb                   # Double DQN (DDQN) implementation
├── Dueling DQN.ipynb                  # Dueling DQN implementation
├── DQN_Soft Update.ipynb              # DQN with Soft Update target network
├── (HW3-1)DQN_Evolution_Report.pdf
├── (HW3-2)RL_Comprehensive_Analysis_V3.pdf
├── (HW3-3)DQN_PyTorch_vs_Keras_Comparison.pdf
└── (HW3-3(bonus))DQN_Soft_Update_Analysis_Report.pdf
```

---

## HW3-1: DQN Evolution Report
[(HW3-1)DQN_Evolution_Report.pdf](https://github.com/user-attachments/files/27430162/HW3-1.DQN_Evolution_Report.pdf)
> **Source**: `(HW3-1)DQN_Evolution_Report.pdf`  
> **Reference**: Chapter 3 — Listing 3.3, 3.5, 3.7

### Overview

This report examines three progressive versions of DQN implemented in `DQN.ipynb`, analyzing how each architectural improvement stabilizes training and reduces loss.

---

### Listing 3.3 — Naïve DQN

- **Mechanism**: Pure online learning — the agent updates its weights after every step without any memory or separate target.
- **Loss Behavior**: Loss fluctuates wildly, often exceeding **1400**, with no convergence.
- **Problem**: Highly correlated consecutive samples cause the gradient updates to reinforce instabilities, preventing stable Q-value estimation.

### Listing 3.5 — DQN + Experience Replay

**Key Additions:**
- Introduced `replay = deque()` to store past experiences.
- Samples a random mini-batch via `random.sample(replay, batch_size)` to break temporal correlations.
- Enforces i.i.d. (independent and identically distributed) training samples.

**Result**: Loss drops significantly to approximately **~250**, showing early convergence behavior.

### Listing 3.7 — DQN + Experience Replay + Target Network

**Key Additions:**
- Introduced a separate **target network**: `model2 = copy.deepcopy(model)`
- TD Target is computed using `model2` (target network), while `model` (online network) is trained.
- The target network is periodically updated: `model2.load_state_dict(model.state_dict())`

**Result**: Loss stabilizes to approximately **~14**, showing smooth and reliable convergence.

---

### Loss Comparison Summary

| Version | Key Technique | Final Loss | Stability |
|---------|---------------|:----------:|:---------:|
| Listing 3.3 | Naïve DQN (no replay, no target) | ~1400+ | ❌ Unstable |
| Listing 3.5 | + Experience Replay | ~250 | ⚠️ Moderate |
| Listing 3.7 | + Target Network | ~14 | ✅ Stable |

**Key Insight**: DQN's power lies in **decoupling** the learning signal. Listing 3.5 separates data collection from training; Listing 3.7 further separates action selection from value evaluation — together reducing loss by **100×** (from ~1400 to ~14).

---

## HW3-2: RL Comprehensive Analysis
[(HW3-2)RL_Comprehensive_Analysis_V3.pdf](https://github.com/user-attachments/files/27430181/HW3-2.RL_Comprehensive_Analysis_V3.pdf)
> **Source**: `(HW3-2)RL_Comprehensive_Analysis_V3.pdf`  
> **Algorithms compared**: DQN · Double DQN · Dueling DQN
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/26e4d539-f175-4911-bb85-0bce1a3edeeb" />
### Performance Summary

| Algorithm | Win Rate | Convergence Speed | Stability |
|-----------|:--------:|:------------------:|:---------:|
| DQN | 90.5% | Slow (high loss oscillation) | ⚠️ Moderate |
| Double DQN | 100.0% | ~7,500 steps | ✅ Stable |
| Dueling DQN | 100.0% | Fastest | ✅ Most Stable |

---

### DQN — Baseline

- Implements standard Q-Learning with neural network approximation.
- Prone to **Q-value overestimation** due to using the same network for both action selection and value evaluation.
- Training loss shows high oscillation especially during early episodes.

---

### Double DQN (DDQN) — Fixing Overestimation

#### Problem with Vanilla DQN
In standard DQN, the same target network is used for both **selecting** the best action and **evaluating** its value. The `max` operator tends to pick overestimated Q-values.

#### DQN TD Target:
```
Y_t = R_{t+1} + γ · max_a Q(S_{t+1}, a; θ⁻)
```

#### Double DQN TD Target:
```
Y_t = R_{t+1} + γ · Q(S_{t+1}, argmax_a Q(S_{t+1}, a; θ); θ⁻)
```

**Decoupling**: The **online network (θ)** selects the action; the **target network (θ⁻)** evaluates it.

**Result**: Loss converges rapidly to ~0 within ~7,500 steps, achieving **100% win rate**.

---

### Dueling DQN — Decomposing Q-Values

#### Architecture
Instead of directly estimating Q(s, a), the network decomposes Q into:
- **V(s)**: State Value — how good is this state overall?
- **A(s, a)**: Advantage — how much better is action *a* compared to others?

#### Dueling DQN Formula:
```
Q(s, a; θ, α, β) = V(s; θ, β) + ( A(s, a; θ, α) - (1/|A|) Σ_{a'} A(s, a'; θ, α) )
```

**Design insight**: When all actions are equally valuable (no dominant action), the advantage terms cancel out, and the network focuses on learning V(s) — making learning more efficient in states where action choice matters less.

**Result**: Achieves **100% win rate** with the fastest and most stable convergence among all three variants.

---

## HW3-3: PyTorch vs. Keras Comparison
[(HW3-3)DQN_PyTorch_vs_Keras_Comparison.pdf](https://github.com/user-attachments/files/27430195/HW3-3.DQN_PyTorch_vs_Keras_Comparison.pdf)
> **Source**: `(HW3-3)DQN_PyTorch_vs_Keras_Comparison.pdf`  
> **Notebooks**: `DQN.ipynb` (PyTorch) vs. `DQN_keras.ipynb` (Keras/TensorFlow)

### Framework Comparison

| Feature | PyTorch (`DQN.ipynb`) | Keras/TF (`DQN_keras.ipynb`) |
|---------|----------------------|------------------------------|
| Execution Mode | Imperative / Eager | Declarative / Graph Mode |
| Layer Definition | `torch.nn.Linear` | `layers.Dense` |
| Tensor Conversion | `torch.from_numpy()` (manual) | `tf.constant()` (automatic) |
| Gradient Computation | `loss.backward()` | `tf.GradientTape` |
| JIT Compilation | Not used | `@tf.function` decorator |
| Inference Call | `model(x)` | `model(x, training=False)` |

---

### 2.1 PyTorch Training Loop

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

PyTorch uses **imperative automatic differentiation** — gradients are computed dynamically and model weights are updated step-by-step.

### 2.2 Keras/TensorFlow Training Loop

```python
with tf.GradientTape() as tape:
    # Forward pass
    gradients = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

Keras uses **`tf.GradientTape`** for explicit gradient tracking, combined with `@tf.function` for JIT (Just-In-Time) compilation to optimize repeated training steps.

---

### Keras-Specific Optimizations

1. **Inference mode**: Use `model(x, training=False)` instead of `model.predict()` for faster single-sample inference.
2. **JIT compilation**: `@tf.function` compiles the training step into a static computation graph, significantly speeding up the training loop.
3. **Target network sync**: Performed every N steps to maintain training stability.

### Results

Both frameworks successfully trained DQN on Gridworld, achieving **>90% win rate**.

- **PyTorch**: More flexible and Pythonic; easier debugging with dynamic graphs.
- **Keras**: More structured; `@tf.function` reduces Python overhead for repeated training steps, leading to potentially faster wall-clock training.

---

## HW3-3 Bonus: DQN Soft Update Analysis
[(HW3-3(bonus))DQN_Soft_Update_Analysis_Report.pdf](https://github.com/user-attachments/files/27430214/HW3-3.bonus.DQN_Soft_Update_Analysis_Report.pdf)

> **Source**: `(HW3-3(bonus))DQN_Soft_Update_Analysis_Report.pdf`  
> **Notebook**: `DQN_Soft Update.ipynb`

### Overview

This bonus report compares two target network update strategies:
- **Hard Update**: Periodic full copy of online network weights to target network
- **Soft Update**: Gradual blending of weights using Polyak Averaging

### Performance Comparison

| Strategy | Update Method | Win Rate | Loss Behavior |
|----------|--------------|:--------:|:-------------:|
| DQN (Hard Update) | Full copy every 500 steps | 90.5% | Occasional spikes |
| DQN Soft Update | Polyak averaging every step | **91.0%** | Smooth & stable |

---

### Why Soft Update?

#### Hard Update (Periodic Replacement)
Every fixed number of steps (e.g., 500), the target network weights are **completely replaced**:
```python
target_model.load_state_dict(online_model.state_dict())
```
**Problem**: The sudden jump in target Q-values can cause instability in the loss function right after each update, adding noise to the training signal.

#### Soft Update (Polyak Averaging)
The target network weights shift **gradually** toward the online network weights at every step:
```
θ_target ← τ · θ_main + (1 - τ) · θ_target
```
Where **τ = 0.001** (only 0.1% of the online network's weights are blended each step).

**Advantages**:
- Eliminates sudden discontinuities in the target Q-values
- The target acts as a **Moving Target** — stable and slowly evolving
- Reduces loss spikes and improves overall training smoothness
- Slight but measurable win-rate improvement: **90.5% → 91.0%**

---

### Mathematical Formulation

```
θ_target ← τ · θ_main + (1 - τ) · θ_target
```

With τ = 0.001:
- Each update only moves the target 0.1% toward the current online network
- Target network effectively tracks a **slow exponential moving average** of the online network

> **Historical Note**: Soft Update was introduced by Google DeepMind in the **DDPG** paper for continuous control tasks and has since been widely adopted for stability.
> 
> *Lillicrap, T. P., et al. (2015). "Continuous control with deep reinforcement learning." arXiv:1509.02971.*

---

## Key Results Summary

| Report | Algorithm | Technique | Win Rate | Final Loss |
|--------|-----------|-----------|:--------:|:----------:|
| HW3-1 | Naïve DQN | Online only | — | ~1400 |
| HW3-1 | DQN + Replay | Experience Replay | — | ~250 |
| HW3-1 | Full DQN | Replay + Target Network | — | ~14 |
| HW3-2 | DQN | Baseline | 90.5% | Oscillating |
| HW3-2 | Double DQN | Decoupled selection/eval | 100.0% | ~0 (7.5k steps) |
| HW3-2 | Dueling DQN | V(s) + A(s,a) decomposition | 100.0% | ~0 (fastest) |
| HW3-3 | DQN (PyTorch) | Baseline | >90% | Stable |
| HW3-3 | DQN (Keras) | @tf.function + GradientTape | >90% | Stable |
| Bonus | DQN Hard Update | Copy every 500 steps | 90.5% | Occasional spikes |
| Bonus | DQN Soft Update | Polyak (τ=0.001) | **91.0%** | Smooth |

---

## References

1. **DQN**: Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). *"Human-level control through deep reinforcement learning."* Nature, 518(7540), 529–533.

2. **Double DQN**: van Hasselt, H., Guez, A., & Silver, D. (2016). *"Deep Reinforcement Learning with Double Q-learning."* Proceedings of the AAAI Conference on Artificial Intelligence, 30(1).

3. **Dueling DQN**: Wang, Z., Schaul, T., Hessel, M., et al. (2016). *"Dueling Network Architectures for Deep Reinforcement Learning."* International Conference on Machine Learning (ICML).

4. **DDPG / Soft Update**: Lillicrap, T. P., et al. (2015). *"Continuous control with deep reinforcement learning."* arXiv:1509.02971.

---

*RL Homework 3 | 2026-05-05 | Deep Q-Network Variants*
