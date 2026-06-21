# Forex DDQN Trading Bot

A Double Deep Q-Network (DDQN) reinforcement learning agent that trades forex using a windowed time-series CNN architecture.

## Files

- **`Environment.py`** — `forex` class: a custom Gym-style environment simulating single-position forex trading.
- **`ddqn_keras.py`** — `ReplayBuffer`, `build_dqn`, `DDQNAgent`: experience replay, the CNN model, and the DDQN training/inference logic (Keras).

## Environment (`forex`)

Loads `Indicator_train_date_scale.csv` and steps through it bar-by-bar with a sliding window of `40` candles.

**State**: shape `(17, 40, 1)` — 16 market/indicator features (Hour, Weekday, Month, OHLCV, EMA 9/21/200, RSI 14, Bollinger bands, Williams %R) stacked with a 17th channel encoding current position (`opened_trade`).

**Action**: one-hot vector of length 5:
| idx | meaning |
|---|---|
| 0 | open short |
| 1 | open long |
| 2 | hold/no-op |
| 3 | close short |
| 4 | close long |

**Reward**:
- `+1` for a valid open, `-10` for an invalid action (e.g. opening while already in a trade, or closing with no position open)
- closing a position returns realized P&L: `(close - open) * Lot` for longs, `(open - close) * Lot` for shorts
- `-20` penalty if an episode ends with a position still open (forced unwind)

**Episode end**: triggered when the data hits `Hour==1 & Minute==1` (i.e. a daily boundary) or when the dataset is exhausted.

Key methods:
- `step(action)` → `(observation, reward, done)`
- `episode_reset()` → initial observation, clears position state
- `reset_data()` → rewinds the data cursor to 0

## Agent (`ddqn_keras.py`)

- **`ReplayBuffer`**: fixed-size circular buffer storing `(17,40,1)` states, actions, rewards, next states, and terminal flags.
- **`build_dqn(lr, window=40)`**: deep Conv2D tower (asymmetric `(1,k)`/`(k,1)` separable-style convs) branching into two heads — `out1` (5-way softmax over actions) and `out2` (scalar, intended as lot-size regression).
- **`DDQNAgent`**: standard Double DQN — `q_eval` selects the action, `q_target` (periodically synced via `update_network_parameters`) evaluates it, decoupling action selection from value estimation to reduce overestimation bias.

## Known issues (as currently written)

These will throw or misbehave if run as-is:
- `choose_action` references `action1, action2` but only ever defines `action` — will raise `NameError`.
- `build_dqn`'s `compile(loss={'Action':..., 'n_LOT':...})` doesn't match the actual output layer names (`out1`/`out2`) — will raise a `KeyError`/`ValueError` in Keras.
- `Adam(lr=lr)` is deprecated in newer Keras/TF (use `learning_rate=`).
- No `gym`-style `reset()`/`action_space`/`observation_space` interface — integration with standard RL libraries (e.g. Stable-Baselines3) would need an adapter.

## Requirements

```
numpy
pandas
keras / tensorflow
```

## Usage sketch

```python
from Environment import forex
from ddqn_keras import DDQNAgent

env = forex()
agent = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=5, epsilon=1.0,
                   batch_size=64, input_dims=(17, 40, 1))

state = env.episode_reset()
done = False
while not done:
    action = agent.choose_action(state)
    next_state, reward, done = env.step(action)
    agent.remember(state, action, reward, next_state, done)
    agent.learn()
    state = next_state
```
