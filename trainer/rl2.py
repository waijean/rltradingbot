import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
import argparse
import pickle
import pandas as pd

from sklearn.preprocessing import StandardScaler

from .agent import DQNAgent
from .environment import MultiStockEnv
from .utils import maybe_make_dir, get_data

# task.py arguments
epsilon_decay=None
learning_rate=None
gamma=None
momentum=None
traindata=None
job_dir=None


def get_scaler(env):
    # return scikit-learn scaler object to scale the states
    # Note: you could also populate the replay buffer here

    states = []
    for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


def play_one_episode(agent, env, is_train, scaler):
    # note: after transforming states are already 1xD
    state = env.reset()
    state = scaler.transform([state])
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = scaler.transform([next_state])
        if is_train == "train":
            agent.train(state, action, reward, next_state, done)
        state = next_state

    return info["cur_val"]


def run(mode, episodes):
    # config
    #models_folder = "linear_rl_trader_models"
    #rewards_folder = "linear_rl_trader_rewards"
    #num_episodes = 100
    #batch_size = 32
    initial_investment = 20000

    #maybe_make_dir(models_folder)
    #maybe_make_dir(rewards_folder)

    data = get_data(traindata)
    n_timesteps, n_stocks = data.shape

    n_train = n_timesteps // 2

    train_data = data[:n_train]
    test_data = data[n_train:]

    env = MultiStockEnv(train_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size, gamma, epsilon_decay, momentum, learning_rate)
    scaler = get_scaler(env)
    num_episodes = episodes
    # store the final value of the portfolio (end of episode)
    portfolio_value = []

    if mode == "test":
        # then load the previous scaler
        with open(f"{job_dir}/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        # remake the env with test data
        env = MultiStockEnv(test_data, initial_investment)

        # make sure epsilon is not 1!
        # no need to run multiple episodes if epsilon = 0, it's deterministic
        agent.epsilon = 0.01

        # load trained weights
        agent.load(f"{job_dir}/linear.npz")

    # play the game num_episodes times
    for e in range(num_episodes):
        t0 = datetime.now()
        val = play_one_episode(agent, env, mode, scaler)
        dt = datetime.now() - t0
        print(
            f"episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}"
        )
        portfolio_value.append(val)  # append episode end portfolio value

    # save the weights when we are done
    if mode == "train":
        # save the DQN
        #agent.save(f"{job_dir}/linear.npz")
        agent.save(f"{job_dir}/linear.csv")

        # save the scaler
        #with open(f"{job_dir}/scaler.pkl", "wb") as f:
        #    pickle.dump(scaler, f)

        # plot losses
        plt.plot(agent.model.losses)
        plt.title('Model Losses')
        #plt.show()
        #plt.savefig(f"{job_dir}/model_losses.png")

    # save portfolio value for each episode
    #np.save(f"{job_dir}/{mode}.npy", portfolio_value)
    plt.plot(portfolio_value)
    plt.title('Portfolio value of episodes')
    plt.show()
    pd.DataFrame(portfolio_value).to_csv(f"{job_dir}/linear.csv")
    plt.savefig(f"{job_dir}/portfolio_{mode}_e{epsilon_decay}_l{learning_rate}"
                f"_m{momentum}_g{gamma}.png")

    plt.hist(portfolio_value, bins=10)
    plt.show()
    plt.savefig(f"{job_dir}/portfoliohist_{mode}_e{epsilon_decay}_l{learning_rate}"
                f"_m{momentum}_g{gamma}.png")