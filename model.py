# Reuse imports and constants
from collections import defaultdict
from blackjack import play_blackjack
import random
import numpy as np
import pandas as pd
from typing import Callable, Dict, Tuple, List, Union

# Define action space and state space
ACTIONS: List[str] = ["hit", "stand", "double"]
LEARNING_RATE: float = 0.05
NUM_EPISODES: int = 1000000

# Define ranges for player hand totals and dealer upcards
PLAYER_TOTALS: List[int] = list(range(4, 22))[::-1]
DEALER_CARDS: List[int] = list(range(1, 11))

# Initialize policy table
policy: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(lambda: {"hit": 1/3, "stand": 1/3, "double": 1/3})

# Define decision function using current policy
def model_policy(player_total: int, dealer_upcard: int, can_double: bool) -> str:
    state: Tuple[int, int] = (player_total, dealer_upcard)
    probs: Dict[str, float] = policy[state].copy()

    if player_total >= 20:
        return "stand"  # NEVER hit on 20 or 21

    if not can_double:
        probs["double"] = 0.0

    # Normalize
    total: float = sum(probs.values())
    if total == 0:
        probs = {a: 1/2 if a != "double" else 0.0 for a in ACTIONS}
    else:
        for a in probs:
            probs[a] = probs[a] / total

    return random.choices(ACTIONS, weights=[probs[a] for a in ACTIONS])[0]

# Update function
def update_policy(policy: Dict[Tuple[int, int], Dict[str, float]], state: Tuple[int, int], action: str, result: str) -> None:
    delta: float = 0.0
    if result == "win":
        delta = LEARNING_RATE * (2 if action == "double" else 1)
    elif result == "loss":
        delta = -LEARNING_RATE

    policy[state][action] += delta
    total: float = sum(max(policy[state][a], 0.01) for a in ACTIONS)
    for a in ACTIONS:
        policy[state][a] = max(policy[state][a], 0.01) / total

# Training loop using the real simulator
for _ in range(NUM_EPISODES):
    history: Dict[str, Union[tuple[int, int], str]] = {"state": None, "action": None}

    def logging_policy(player_total: int, dealer_upcard: int, can_double: bool) -> str:
        state = (player_total, dealer_upcard)
        action = model_policy(player_total, dealer_upcard, can_double)
        history["state"] = state
        history["action"] = action
        return action

    result: str = play_blackjack(logging_policy)
    update_policy(policy, history["state"], history["action"], result)

# Build final action matrix for display
action_map: np.ndarray = np.empty((len(PLAYER_TOTALS), len(DEALER_CARDS)), dtype=int)
for i, p in enumerate(PLAYER_TOTALS):
    for j, d in enumerate(DEALER_CARDS):
        state = (p, d)
        best_action: str = max(policy[state], key=policy[state].get)
        action_map[i, j] = ACTIONS.index(best_action)

# Display result
action_labels: List[List[str]] = [[
    ACTIONS[action_map[i][j]] for j in range(len(DEALER_CARDS))
] for i in range(len(PLAYER_TOTALS))]

print("Player Total\tDealer Upcard")

df: pd.DataFrame = pd.DataFrame(
    action_labels,
    index=[f"{p}" for p in PLAYER_TOTALS],
    columns=[f"{d}" for d in DEALER_CARDS]
)

print(df)

df.to_csv("blackjack_strategy.csv")


plot_color_visualization = True
if plot_color_visualization:
    import matplotlib.pyplot as plt

    # Convert action_map to string labels for heatmap plotting
    label_map = np.vectorize(lambda x: ACTIONS[x])(action_map)

    # Create a color-coded numeric map for plotting
    color_map = {"hit": 0, "stand": 1, "double": 2}
    numeric_map = np.vectorize(lambda x: color_map[x])(label_map)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(numeric_map, cmap="viridis", aspect="auto")

    # Axis ticks and labels
    ax.set_xticks(range(len(DEALER_CARDS)))
    ax.set_xticklabels([str(d) for d in DEALER_CARDS])
    ax.set_yticks(range(len(PLAYER_TOTALS)))
    ax.set_yticklabels([str(p) for p in PLAYER_TOTALS])

    # Axis titles
    ax.set_xlabel("Dealer Upcard")
    ax.set_ylabel("Player Total")
    plt.title("Blackjack Strategy Heatmap (0=Hit, 1=Stand, 2=Double)")

    # Add colorbar
    cbar = plt.colorbar(cax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Hit', 'Stand', 'Double'])

    plt.tight_layout()
    plt.show()


plot_confidence = False
if plot_confidence:

    #############################

    import matplotlib.pyplot as plt

    ##############

    # Create confidence matrix (max probability per state)
    max_probs = np.zeros((len(PLAYER_TOTALS), len(DEALER_CARDS)))
    for i, p in enumerate(PLAYER_TOTALS):
        for j, d in enumerate(DEALER_CARDS):
            state = (p, d)
            max_prob = max(policy[state].values())
            max_probs[i, j] = max_prob

    # Plot confidence heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(max_probs, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)

    # Axis labels
    ax.set_xticks(range(len(DEALER_CARDS)))
    ax.set_xticklabels([str(d) for d in DEALER_CARDS])
    ax.set_yticks(range(len(PLAYER_TOTALS)))
    ax.set_yticklabels([str(p) for p in PLAYER_TOTALS])
    ax.set_xlabel("Dealer Upcard")
    ax.set_ylabel("Player Total")
    plt.title("Model Confidence in Best Action (Max Probability)")

    # Colorbar
    cbar = plt.colorbar(cax)
    cbar.ax.set_ylabel("Confidence (%)")
    cbar.ax.set_yticklabels([f"{int(t*100)}%" for t in cbar.get_ticks()])

    plt.tight_layout()
    plt.show()