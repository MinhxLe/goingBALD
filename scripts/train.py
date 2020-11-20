import json
import argparse
import os
from bald.constants import (
    CONLL_DATA_PROCESSED_DIR,
    EXPERIMENTS_RESULT_DIR,
)
from bald.conll_experiment_manager import (
    DropoutBALDExperimentManager,
    MNLPExperimentManager,
    RandomExperimentManager,
    RLExperimentManager
)
from datetime import datetime

parser = argparse.ArgumentParser()
# model args
parser.add_argument('--word_embedding_fname', type=str,
                    default=os.path.join(CONLL_DATA_PROCESSED_DIR, "word2vec.vector.npy"),
                    help='batch size (default: )')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (default: 0.5)')
parser.add_argument('--emb_dropout', type=float, default=0.25,
                    help='dropout applied to the embedded layer (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.35,
                    help='gradient clip, -1 means no clip (default: 0.35)')
parser.add_argument('--char_kernel_size', type=int, default=3,
                    help='character-level kernel size (default: 3)')
parser.add_argument('--word_kernel_size', type=int, default=3,
                    help='word-level kernel size (default: 3)')
parser.add_argument('--emsize', type=int, default=50,
                    help='size of character embeddings (default: 50)')
parser.add_argument('--char_layers', type=int, default=3,
                    help='# of character-level convolution layers (default: 3)')
parser.add_argument('--word_layers', type=int, default=3,
                    help='# of word-level convolution layers (default: 3)')
parser.add_argument('--char_nhid', type=int, default=50,
                    help='number of hidden units per character-level convolution layer (default: 50)')
parser.add_argument('--word_nhid', type=int, default=300,
                    help='number of hidden units per word-level convolution layer (default: 300)')

# training args
parser.add_argument('--train_epochs', type=int, default=1,
                    help='upper training epoch limit (default: 2)')
parser.add_argument('--lr', type=float, default=4,
                    help='initial learning rate (default: 4)')
# parser.add_argument('--optim', type=str, default='SGD',
#                     help='optimizer type (default: SGD)')
parser.add_argument('--weight', type=float, default=10,
                    help='manual rescaling weight given to each tag except "O"')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
# AL args
parser.add_argument('--al_epochs', type=int, default=50,
                    help='# of active learning steps (default: 100)')
parser.add_argument('--rl_epsilon', type=float, default=0.01,
                    help='epsilon used for e greedy bandit')

# experiment logging/debugging
parser.add_argument('--al_sampler', type=str,
                    help='active learning sampler')
# parser.add_argument('--experiment_name', type=str,
#                     default=f'conll2003_random_sampler',
# parser.add_argument('--experiment_name', type=str,
#                     default=f'conll2003_mnlp_sampler_{timestamp_str}',
#                     help='experiment name')
parser.add_argument('--save_interval', type=int, default=10, metavar='N',
                    help='save interval (default: 10)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='report interval (default: 10)')
parser.add_argument('--debug', type=bool, default=False,
                    help='is debug runs on smaller dataset (defaults to False)')
parser.add_argument('--stdout', type=bool, default=False,
                    help='print log message in stdout')

parser.add_argument('--mc_sample_size', type=int, default=100,
                    help='mc sample size for dropout BALD')
args = parser.parse_args()

# TODO add other arguments in
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
args.experiment_name = f"conll2003_{args.al_sampler}_sampler_{timestamp_str}"

if args.debug:
    args.experiment_name = f"conll2003_{args.al_sampler}_sampler_DEBUG"
    args.train_epochs = 1
    args.batch_size = 25
    args.al_epochs = 2
if args.al_sampler == "mnlp":
    manager = MNLPExperimentManager(args, save_exp=True)
elif args.al_sampler == "random":
    manager = RandomExperimentManager(args, save_exp=True)
elif args.al_sampler == "BALD":
    manager = DropoutBALDExperimentManager(args, save_exp=True)
elif args.al_sampler == "e_greedy_bandit":
    manager = RLExperimentManager(args, save_exp=True)
else:
    raise Exception("sampler not implemented")

with open(os.path.join(manager.experiment_dir, "experiment_args.txt"), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
manager.run_experiment()


