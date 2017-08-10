import gtnlplib.parsing as parsing
import gtnlplib.data_tools as data_tools
import gtnlplib.constants as consts
import gtnlplib.evaluation as evaluation
import gtnlplib.utils as utils
import gtnlplib.feat_extractors as feat_extractors
import gtnlplib.neural_net as neural_net

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import time
import os

from collections import defaultdict


# Read in the dataset
dataset = data_tools.Dataset(consts.TRAIN_FILE, consts.DEV_FILE, consts.TEST_FILE)

# Assign each word a unique index, including the two special tokens
word_to_ix = { word: i for i, word in enumerate(dataset.vocab) }

# Some constants to keep around
LSTM_NUM_LAYERS = 1
TEST_EMBEDDING_DIM = 5
WORD_EMBEDDING_DIM = 10
STACK_EMBEDDING_DIM = 300
NUM_FEATURES = 3

# Hyperparameters
ETA_0 = 0.001
DROPOUT = 0.1

torch.manual_seed(1)
feat_extractor = feat_extractors.SimpleFeatureExtractor()
word_embedding_lookup = neural_net.VanillaWordEmbeddingLookup(word_to_ix, STACK_EMBEDDING_DIM)
# if os.path.isfile(os.getcwd()+"/Checkpoints/action_chooser"):
#     action_chooser = torch.load(os.getcwd()+"/Checkpoints/action_chooser")
# else:
#     action_chooser = neural_net.ActionChooserNetwork(STACK_EMBEDDING_DIM * NUM_FEATURES)
# if os.path.isfile(os.getcwd()+"/Checkpoints/combiner_network"):
#     combiner_network = torch.load(os.getcwd()+"/Checkpoints/combiner_network")
# else:
#     combiner_network = neural_net.MLPCombinerNetwork(STACK_EMBEDDING_DIM)
# parser = parsing.TransitionParser(feat_extractor, word_embedding_lookup, action_chooser, combiner_network)
if os.path.isfile(os.getcwd()+"/Checkpoints/parser_dict"):
    parser.load_state_dict(torch.load(os.getcwd()+"/Checkpoints/parser_dict"))
parser = torch.load(os.getcwd()+"/Checkpoints/parser")

# parsing.evaluate(dataset.dev_data[0:100], parser, verbose=True, prob=True)
# dev_sentences = [ sentence for sentence, _ in dataset.dev_data ]
# evaluation.output_preds("dev_st_v1.parse", parser, dev_sentences)
evaluation.output_preds("test_st_v0.parse", parser, dataset.test_data)
