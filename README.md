# Space Wars

## Overview

Space Wars is an exciting space-themed game where players engage in epic battles against an AI agent trained using Q-learning. The game offers an immersive experience with challenging gameplay, allowing players to test their skills against an intelligent opponent.

## Features

- **Single Player Mode**: Battle against a Reinforcement Learning (RL) agent.
- **AI Agent**: The AI opponent is trained using Q-learning to provide a challenging gameplay experience.
- **Dynamic Gameplay**: Fast-paced and engaging space battles.
- **Graphics and Sound**: Enhanced with captivating visuals and sound effects.

## Installation

To get started with Space Wars, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/space-wars.git
    cd space-wars
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the game**:
    ```bash
    python main.py
    ```

## Gameplay Instructions

1. **Start the Game**: Launch the game by running the `main.py` file.
2. **Control Your Spaceship**: Use the arrow keys to navigate your spaceship and the spacebar to fire.
3. **Objective**: Destroy the AI opponent's spaceship while avoiding being hit.
4. **Scoring**: Points are awarded for each successful hit on the opponent's spaceship.

## AI Agent

The AI opponent is trained using Q-learning, a type of reinforcement learning algorithm. The agent learns to play the game by receiving rewards for positive actions and penalties for negative actions, enabling it to develop a strategy to defeat the player.

## Development

### Prerequisites

- Python 3.8 or higher
- Pygame
- NumPy

### Training the AI

If you wish to train the AI agent yourself, follow these steps:

1. **Train the agent**:
    ```bash
    python train.py
    ```

2. **Evaluate the agent**:
    ```bash
    python evaluate.py
    ```

## Contributing

We welcome contributions to improve Space Wars! If you have any suggestions, bug reports, or want to contribute code, please create an issue or submit a pull request.

## Acknowledgements

- The Pygame community for providing excellent resources and support.
- OpenAI for inspiration on reinforcement learning techniques.

---

Enjoy playing Space Wars and challenging the AI agent!
