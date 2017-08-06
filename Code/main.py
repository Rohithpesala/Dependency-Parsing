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

# g - seperate word embeddings trained on WSJ, test on GENIA
# 1 - joint WE trained on GENIA, test on GENIA
# ""- joint WE trained on WSJ, test on GENIA

torch.manual_seed(1)
feat_extractor = feat_extractors.SimpleFeatureExtractor()
word_embedding_lookup = neural_net.VanillaWordEmbeddingLookup(word_to_ix, STACK_EMBEDDING_DIM)
if os.path.isfile(os.getcwd()+"/Checkpoints/action_chooser"):
    action_chooser = torch.load(os.getcwd()+"/Checkpoints/action_chooser")
else:
    action_chooser = neural_net.ActionChooserNetwork(STACK_EMBEDDING_DIM * NUM_FEATURES)
if os.path.isfile(os.getcwd()+"/Checkpoints/combiner_network"):
    combiner_network = torch.load(os.getcwd()+"/Checkpoints/combiner_network")
else:
    combiner_network = neural_net.MLPCombinerNetwork(STACK_EMBEDDING_DIM)

parser = parsing.TransitionParser(feat_extractor, word_embedding_lookup, action_chooser, combiner_network)
# parser = torch.load(os.getcwd()+"/Checkpoints/parser")
if os.path.isfile(os.getcwd()+"/Checkpoints/parser_dict_st_v2"):
    parser.load_state_dict(torch.load(os.getcwd()+"/Checkpoints/parser_dict_st_v2"))
# else:
#     parser = parsing.TransitionParser(feat_extractor, word_embedding_lookup, action_chooser, combiner_network)

# torch.save(parser.state_dict(),os.getcwd()+"/Checkpoints/parser_dict")
parameters = filter(lambda p: p.requires_grad, parser.parameters())
optimizer = optim.SGD(parameters, lr=ETA_0)

# train the thing for a while here.
# Shouldn't take too long, even on a laptop
f = open("acc_st_v2.txt","w")
start_time = time.time()
for epoch in xrange(7):
    parser.to_cuda()
    print "Epoch {}".format(epoch+1)
    for i in range(478):
        print i
        parsing.train(dataset.training_data[(i*100):(i+1)*100], parser, optimizer, verbose=True)
        if i%100==0:
            torch.save(parser.state_dict(),os.getcwd()+"/Checkpoints/parser_dict_st_v2")
            torch.save(action_chooser,os.getcwd()+"/Checkpoints/action_chooser")
            torch.save(combiner_network,os.getcwd()+"/Checkpoints/combiner_network")
    
    print "Dev Evaluation"
    parser.to_cpu()
    pacc,ploss = parsing.evaluate(dataset.dev_data[0:100], parser, verbose=True)
    print "F-Score: {}".format(evaluation.compute_metric(parser, dataset.dev_data[0:100], evaluation.fscore))
    # print "Attachment Score: {}".format(evaluation.compute_attachment(parser, dataset.dev_data[0:100]))
    print "\n"
    
    f.write(str(epoch+1))
    f.write("\n")
    f.write("Accuracy:"+str(pacc)+"\t"+"loss:"+str(ploss)+"\n")
    torch.save(parser.state_dict(),os.getcwd()+"/Checkpoints/parser_dict_st_v2")
    torch.save(action_chooser,os.getcwd()+"/Checkpoints/action_chooser")
    torch.save(combiner_network,os.getcwd()+"/Checkpoints/combiner_network")
print time.time()-start_time
