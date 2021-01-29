

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

train_rewards = np.load(
    "linear_rl_trader_rewards/train_v10_6_w_tech_indicators.npy")
test_rewards = np.load(
    "linear_rl_trader_rewards/test_v10_6_w_tech_indicators.npy")
random_rewards = np.load(
    "linear_rl_trader_rewards/random_vnn_5_w_tech_indicators.npy")

rewards_df = pd.DataFrame({"train":train_rewards, "test":test_rewards, "random":random_rewards})

rewards_df.describe()

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20,5))
rewards_df["train"].plot(ax=axes[0])
axes[0].set_title('Portfolio value (train)')
rewards_df["test"].plot(ax=axes[1])
axes[1].set_title('Portfolio value (test)')
rewards_df["random"].plot(ax=axes[2])
axes[2].set_title('Portfolio value (random)')

sns.displot(rewards_df)

#############################

no_tech_train_rewards = np.load(
    "linear_rl_trader_rewards/train_vnn_5a_no_tech_indicators.npy")
no_tech_test_rewards = np.load(
    "linear_rl_trader_rewards/test_vnn_5a_no_tech_indicators.npy")
no_tech_random_rewards = np.load(
    "linear_rl_trader_rewards/random_vnn_5_w_tech_indicators.npy")

no_tech_rewards_df = pd.DataFrame(
    {"train":no_tech_train_rewards, "test":no_tech_test_rewards, "random":no_tech_random_rewards})

no_tech_rewards_df.describe()

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20,5))
no_tech_rewards_df["train"].plot(ax=axes[0])
axes[0].set_title('Portfolio value (train)')
no_tech_rewards_df["test"].plot(ax=axes[1])
axes[1].set_title('Portfolio value (test)')
no_tech_rewards_df["random"].plot(ax=axes[2])
axes[2].set_title('Portfolio value (random)')

sns.displot(no_tech_rewards_df)

