import argparse
from . import rl2

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--mode", type=str, required=True, help='either "train" or "test"'
)
parser.add_argument(
    "-e", "--epsilondecay", type=float, required=True, help='rate of epsilon decay'
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
parser.add_argument(
    "-j", "--job-dir", type=str, required=True, help='job directory'
)
parser.add_argument(
    "-d", "--traindata", type=str, required=True, help='training data'
)
args = parser.parse_args()

rl2.learning_rate = args.learnrate
rl2.gamma = args.gamma
rl2.momentum = args.momentum
rl2.epsilon_decay = args.epsilondecay
rl2.traindata = args.traindata
rl2.job_dir=args.job_dir

#rl2.run(args.mode, args.episodes)
rl2.run("train", 100)