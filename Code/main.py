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
DROPOUT = 0.0

torch.manual_seed(1)
feat_extractor = feat_extractors.SimpleFeatureExtractor()
word_embedding_lookup = neural_net.VanillaWordEmbeddingLookup(word_to_ix, STACK_EMBEDDING_DIM)
action_chooser = neural_net.ActionChooserNetwork(STACK_EMBEDDING_DIM * NUM_FEATURES)
combiner_network = neural_net.MLPCombinerNetwork(STACK_EMBEDDING_DIM)
parser = parsing.TransitionParser(feat_extractor, word_embedding_lookup,
                                     action_chooser, combiner_network)
optimizer = optim.SGD(parser.parameters(), lr=ETA_0)

# train the thing for a while here.
# Shouldn't take too long, even on a laptop
for epoch in xrange(1):
    print "Epoch {}".format(epoch+1)
    for i in range(1):
        print i
        parsing.train(dataset.training_data[(i*100):(i+1)*100], parser, optimizer, verbose=True)
    
    print "Dev Evaluation"
    parsing.evaluate(dataset.dev_data[0:100], parser, verbose=True)
    print "F-Score: {}".format(evaluation.compute_metric(parser, dataset.dev_data[0:100], evaluation.fscore))
    print "Attachment Score: {}".format(evaluation.compute_attachment(parser, dataset.dev_data[0:100]))
    print "\n"