import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
import argparse
import pickle

from sklearn.preprocessing import StandardScaler

from agent import DQNAgent
from environment import MultiStockEnv
from utils import maybe_make_dir, get_data, get_data_from_yf


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
    version = "1_no_tech_indicators"
    models_folder = "linear_rl_trader_models"
    rewards_folder = "linear_rl_trader_rewards"
    num_episodes = 2000
    batch_size = 32
    initial_investment = 20000
    n_stocks = 3
    n_inds = 0
    is_deep_learning = False

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--mode", type=str, required=True, help='either "train" or "test" or "random"'
    )
    args = parser.parse_args()

    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)

    data = get_data_from_yf(["AAPL","MSI","SBUX"], is_tech_ind=False)
    n_timesteps, cols = data.shape
    assert cols == n_stocks + n_inds, f"Expected {n_stocks + n_inds} but Actual {cols}"

    n_train = n_timesteps // 2

    train_data = data[:n_train]
    test_data = data[n_train:]

    env = MultiStockEnv(stock_price_history=train_data[:,:n_stocks], technical_ind=train_data[:, n_stocks:], initial_investment=initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    print(f"State size: {state_size}, Action size: {action_size}")
    agent = DQNAgent(state_size, action_size, is_deep_learning=is_deep_learning)
    scaler = get_scaler(env)

    # store the final value of the portfolio (end of episode)
    portfolio_value = []
    
    if args.mode == "random":
        # remake the env with test data
        env = MultiStockEnv(stock_price_history=test_data[:,:n_stocks], technical_ind=test_data[:, n_stocks:], initial_investment=initial_investment)

        # set epsilon to 1 so it's always in exploration
        agent.epsilon = 1

    if args.mode == "test":
        # then load the previous scaler
        with open(f"{models_folder}/scaler_v{version}.pkl", "rb") as f:
            scaler = pickle.load(f)

        # remake the env with test data
        env = MultiStockEnv(stock_price_history=test_data[:,:n_stocks], technical_ind=test_data[:, n_stocks:], initial_investment=initial_investment)

        # make sure epsilon is not 1!
        # no need to run multiple episodes if epsilon = 0, it's deterministic
        agent.epsilon = 0.01

        # load trained weights
        agent.load(f"{models_folder}/linear_v{version}.npz")

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
        agent.save(f"{models_folder}/linear_v{version}.npz")

        # save the scaler
        with open(f"{models_folder}/scaler_v{version}.pkl", "wb") as f:
            pickle.dump(scaler, f)

        # plot losses
        plt.plot(agent.model.losses)
        plt.show()

    # save portfolio value for each episode
    np.save(f"{rewards_folder}/{args.mode}_v{version}.npy", portfolio_value)
