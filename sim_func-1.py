import numpy as np
import matplotlib.pyplot as plt

def simulate(k=200, D=np.array([1, 0, 0, 0, 0, 0]), delta=0, warmup=False):
    """
    Simulates a sequence of states and observations for given HMM parameters.

    Args:
    - k: Length of the sequence (default 200)
    - D: Initial state probabilities (default [1, 0, 0, 0, 0, 0])
    - delta: Policy modifier (default 0)
    - warmup: If True, simulate 50 warm-up steps and return the result from step 51 (default False)

    Returns:
    - s_onehot: One-hot encoded state sequence
    - o_onehot: One-hot encoded observation sequence
    """

    B = np.array([
        [0.1, 0.9, 0, 0, 0, 0],
        [0.2, 0, 0.1 + delta, 0, 0, 0.7 - delta],
        [0, 0.5, 0, 0.5, 0, 0],
        [0, 0, 0.5, 0.5, 0, 0],
        [0, 0, 0.1, 0, 0.4, 0.5],
        [0, 0.2, 0, 0, 0.7, 0.1]
    ])

    A = np.array([
        [0.25, 0.25, 0.25, 0.25],
        [0.4, 0.4, 0.1, 0.1],
        [0.4, 0.4, 0.1, 0.1],
        [0.3, 0.3, 0.3, 0.1],
        [0.1, 0.1, 0.2, 0.6],
        [0.15, 0.1, 0.35, 0.4]
    ])

    total_steps = k + 50 if warmup else k

    s = np.zeros(total_steps, dtype=int)
    o = np.zeros(total_steps, dtype=int)

    # Initial state and observation
    s[0] = np.random.choice(6, p=D)
    o[0] = np.random.choice(4, p=A[s[0], :])

    # Simulate the sequence
    for t in range(1, total_steps):
        s[t] = np.random.choice(6, p=B[s[t - 1], :])
        o[t] = np.random.choice(4, p=A[s[t], :])

    # Apply warm-up logic
    if warmup:
        s = s[50:]
        o = o[50:]

    s_onehot = np.eye(6)[s]
    o_onehot = np.eye(4)[o]

    return s_onehot, o_onehot

# Example usage
k = 100
D = np.array([1, 0, 0, 0, 0, 0])
delta = 0
warmup = True
states, observations = simulate(k, D, delta,warmup)

# Plotting the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.imshow(states.T, aspect='auto', cmap='viridis')
ax1.set_title('States')
ax1.set_xlabel('Time step')
ax1.set_ylabel('State')
ax1.set_yticks(np.arange(6))
ax1.set_yticklabels(['1', '2', '3', '4', '5', '6'])

ax2.imshow(observations.T, aspect='auto', cmap='plasma')
ax2.set_title('Observations')
ax2.set_xlabel('Time step')
ax2.set_ylabel('Observation')
ax2.set_yticks(np.arange(4))
ax2.set_yticklabels(['α', 'β', 'γ', 'μ'])

plt.tight_layout()
plt.show()