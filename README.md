# Racing Game - Deep Double Q-Network (DDQN) Implementation

## Overview

This project implements a Deep Double Q-Network (DDQN) to train an agent to play the CarRacing-v3 environment from OpenAI Gymnasium. The implementation includes prioritized experience replay, frame skipping, and parallel environments for efficient training.

## Network Architecture

The DDQN network is implemented in `src/models/ddqn_network.py` and consists of the following layers:

1. **Convolutional Layers**:

   - `Conv2d`: Input channels = 1, Output channels = 6, Kernel size = 7, Stride = 3
   - `MaxPool2d`: Kernel size = 2
   - `Conv2d`: Input channels = 6, Output channels = 12, Kernel size = 4
   - `MaxPool2d`: Kernel size = 2

2. **Fully Connected Layers**:
   - `Linear`: Input size = 432, Output size = 216
   - `Linear`: Input size = 216, Output size = 64
   - `Linear`: Input size = 64, Output size = 12 (number of actions)

The network uses ReLU activations for all layers except the output layer.

## Reward Function

The reward function is based on the CarRacing environment's default reward system, with additional penalties:

- **Negative Rewards**: Clipped to a minimum of -10 to discourage poor actions.
- **Grass Penalty**: Incremented when the agent drives on grass.
- **Termination Conditions**:
  - Episode ends if the agent accumulates 25 consecutive negative rewards.
  - Episode ends if the agent spends 20 consecutive steps on grass.
  - Episode ends if the road is no longer visible.

## Loss Function

The loss function is implemented in the `experience_replay` method of the `DDQNAgent` class (`src/agents/ddqn_agent.py`). It uses the Mean Squared Error (MSE) between the predicted Q-values and the target Q-values:

- **Target Q-Value**:
  \[ Q*{target} = r + \gamma \cdot Q*{next} \cdot (1 - done) \]
  where:

  - \( r \): Reward
  - \( \gamma \): Discount factor
  - \( Q\_{next} \): Q-value of the next state from the target network
  - \( done \): Boolean indicating if the episode is finished

- **Temporal Difference (TD) Error**:
  \[ TD*{error} = Q*{current} - Q\_{target} \]

- **Loss**:
  \[ Loss = \text{MSE}(TD\_{error}) \]

## Key Features

1. **Prioritized Replay Buffer**:

   - Stores experiences with priorities based on TD errors.
   - Samples experiences with probabilities proportional to their priorities.
   - Implements importance sampling to correct for sampling bias.

2. **Frame Skipping**:

   - Executes the same action for multiple frames to speed up training.

3. **Parallel Environments**:

   - Supports training with multiple environments simultaneously for faster data collection.

4. **Target Network**:

   - A separate network is used to compute target Q-values, updated periodically to stabilize training.

5. **Epsilon-Greedy Policy**:
   - Balances exploration and exploitation by selecting random actions with probability \( \epsilon \), which decays over time.

## Training

The training process is managed by the `DDQNTrainer` class (`src/training/trainer.py`). Key steps include:

1. Reset the environment and initialize the state.
2. Choose an action using the epsilon-greedy policy.
3. Execute the action and store the transition in the replay buffer.
4. Sample a batch of experiences from the buffer and update the main network using the loss function.
5. Periodically update the target network and save checkpoints.

## Evaluation

The agent is evaluated using the `evaluate_agent` method in `DDQNTrainer`. During evaluation:

- The agent selects actions greedily (without exploration).
- Metrics such as average reward, standard deviation, and episode duration are computed.

## File Structure

- `src/models/ddqn_network.py`: Defines the DDQN network architecture.
- `src/agents/ddqn_agent.py`: Implements the DDQN agent, including action selection, experience replay, and model saving/loading.
- `src/replay_buffer/prioritized_buffer.py`: Implements the prioritized replay buffer.
- `src/training/trainer.py`: Manages training and evaluation.
- `configs/ddqn_config.yaml`: Configuration file for hyperparameters.
- `scripts/train.py`: Script to train the agent.
- `scripts/test.py`: Script to evaluate the agent.

## How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Train the agent:

   ```bash
   python scripts/train.py --config configs/ddqn_config.yaml
   ```

3. Evaluate the agent:
   ```bash
   python scripts/test.py --model models/default/episode_1900.pth
   ```

## Hyperparameters

Default hyperparameters are defined in `DDQNConfig` (`src/utils/config.py`):

- Episodes: 2000
- Batch size: 20
- Learning rate: 0.001
- Gamma: 0.95
- Epsilon: 1.0 (decays to 0.1)
- Replay buffer size: 150,000
- Target update steps: 5
- Frame skip: 2

## Results

The agent achieves consistent performance on the CarRacing-v3 environment, with average rewards improving over training episodes. Checkpoints and reward data are saved periodically for analysis.

## Future Work

- Implement additional reward shaping techniques.
- Experiment with different network architectures.
- Optimize hyperparameters using automated search methods.
- Extend support for other environments.

## Acknowledgments

This project is inspired by the Deep Q-Learning and Double Q-Learning algorithms. Special thanks to OpenAI for the CarRacing environment and the Gymnasium framework.
