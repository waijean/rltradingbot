import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
import argparse
import pickle
import pandas as pd

from sklearn.preprocessing import StandardScaler

from agent import DQNAgent
from environment import MultiStockEnv
from utils import maybe_make_dir, get_data
import seaborn as sns


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


def play_one_episode(agent, env, is_train):
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


if __name__ == "__main__":

    # config
    models_folder = "linear_rl_trader_models"
    rewards_folder = "linear_rl_trader_rewards"
    #num_episodes = 100
    batch_size = 32
    initial_investment = 20000

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--mode", type=str, required=True, help='either "train" or "test"'
    )
    parser.add_argument(
        "-e", "--epsilon-decay", type=float, required=True, help='rate of epsilon decay'
    )
    parser.add_argument(
        "-l", "--learnrate", type=float, required=True, help='learning rate'
    )
    parser.add_argument(
        "-t", "--momentum", type=float, required=True, help='momentum'
    )
    parser.add_argument(
        "-g", "--gamma", type=float, required=True, help='gamma'
    )
    parser.add_argument(
        "-n", "--episodes", type=int, required=True, help='number of episodes'
    )
    args = parser.parse_args()

    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)

    data = get_data()
    n_timesteps, n_stocks = data.shape

    n_train = n_timesteps // 2

    train_data = data[:n_train]
    test_data = data[n_train:]

    env = MultiStockEnv(train_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size, args.gamma, args.epsilon_decay, args.momentum, args.learnrate)
    scaler = get_scaler(env)
    num_episodes = args.episodes

    # store the final value of the portfolio (end of episode)
    portfolio_value = []

    if args.mode == "test":
        # then load the previous scaler
        with open(f"{models_folder}/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        # remake the env with test data
        env = MultiStockEnv(test_data, initial_investment)

        # make sure epsilon is not 1!
        # no need to run multiple episodes if epsilon = 0, it's deterministic
        agent.epsilon = 0.01

        # load trained weights
        agent.load(f"{models_folder}/linear.npz")

    # play the game num_episodes times
    for e in range(num_episodes):
        t0 = datetime.now()
        val = play_one_episode(agent, env, args.mode)
        dt = datetime.now() - t0
        print(
            f"episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}"
        )
        portfolio_value.append(val)  # append episode end portfolio value

    # save the weights when we are done
    if args.mode == "train":
        # save the DQN
        agent.save(f"{models_folder}/linear.npz")

        # save the scaler
        with open(f"{models_folder}/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        # plot losses
        plt.plot(agent.model.losses)
        plt.title('Model Losses')
        plt.show()
        #plt.savefig(f"{models_folder}/model_losses.png")

    # save portfolio value for each episode
    np.save(f"{rewards_folder}/{args.mode}.npy", portfolio_value)
    plt.plot(portfolio_value)
    plt.title('Portfolio value of episodes')
    plt.show()
    plt.savefig(f"{rewards_folder}/portfolio_{args.mode}_e{args.epsilon_decay}_l{args.learnrate}"
                f"_m{args.momentum}_g{args.gamma}.png")

    plt.hist(portfolio_value, bins=10)
    plt.show()
    plt.savefig(f"{rewards_folder}/portfoliohist_{args.mode}_e{args.epsilon_decay}_l{args.learnrate}"
                f"_m{args.momentum}_g{args.gamma}.png")