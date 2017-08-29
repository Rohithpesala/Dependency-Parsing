# import modules & set up logging
import gensim, logging,os
import torch

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()




logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


 
# sentences = [['first', 'sentence'], ['second', 'sentence']]
# sentences = MySentences("data_GENIA")
# model = gensim.models.Word2Vec(sentences, min_count=5, iter = 500, workers=5,size=300)
fname = 'model_GENIA'
# model.save(fname)
model = gensim.models.Word2Vec.load(fname)
model2 = torch.load(os.getcwd()+"/Mod_model_GENIA")

print len((model.wv).vocab)
print len(model2)
j = 0
for i in (model.wv).vocab:
	print i
	# print model.wv[i]
	print type(torch.FloatTensor(model.wv[i]))
	# print model2[i].data[0]
	print type(torch.FloatTensor(model2[i].data[0]))
	break
	j+=1
print j
# print model.wv["SHIFT"]