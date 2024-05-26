# Crypto Trading using RL

This repository contains experiments about cryptocurrency trading using reinforcement learning.  
These experiments are the result of the work published here: [Topical Aspects of Modern Scientific Research, May 2024](https://sci-conf.com.ua/wp-content/uploads/2024/05/TOPICAL-ASPECTS-OF-MODERN-SCIENTIFIC-RESEARCH-16-18.05.2024.pdf) (pages 261-265)

## Methods
All methods used in the experiments are in the `app` folder. 

- `__init__.py`: Initialization file for the app module.
- `data_utils.py`: Utilities for handling and processing data.
- `environment.py`: Definitions and configurations for the trading environment.
- `models_utils.py`: Utility functions for training and testing RL models.
- `render_env.py`: Functions to render the trading environment.
- `visualization.py`: Functions to visualize data and model performance.

## Experiments
All the code for experiments is in the `experiments` folder.  

1. `1_reward_function.py`: Experiment related to defining and testing the reward function to use in the environment.
2. `2_augmented_data.py`: Experiment on using augmented data for training. Augmentation was done using trading indicators.
3. `3_window_size.py`: Experiment to determine the optimal window size for the input data.
4. `4_recurrent_ppo.py`: Experiment with a recurrent version of the Proximal Policy Optimization (PPO) algorithm.
5. `5_hyperparameters_optimization.py`: Experiment focused on optimizing the hyperparameters of the models.
6. `6_final_training.py`: Code for the final training of models.
7. `7_testing.py`: Code for testing the trained models.

## Setup

To set up the environment for running the experiments, follow these steps:

1. **Clone the repository**
    ```sh
    git clone https://github.com/IvanShkvir/crypto-trading-rl-2.git
    cd crypto-trading-rl
    ```

2. **Create a virtual environment**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies**
    ```sh
    pip install -r requirements.txt
    ```

4. **(Optional) Install the reduced and modified set of dependencies using this command if you want to run experiments in Google Colab or similar applications**
    ```sh
    pip install -r requirements-colab.txt
    ```

## Running Experiments

To run a specific experiment, navigate to the `experiments` folder and execute the corresponding script. For example, to run the reward function experiment:
```sh
python experiments/1_reward_function.py
```
To run a specific experiment in Colab convert the experiment file into `.ipynb` and execute it in Google Colab. 
