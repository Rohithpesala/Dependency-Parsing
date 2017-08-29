##############################################
# Non linear transformation from WSJ to GENIA

import torch
import torch.nn as nn
import torch.autograd as ag
import torch.nn.functional as F
from torch import optim

import gensim, logging,os

class MLPTransformation(nn.Module):
	"""
	A non-linear transformation between spaces
	"""

	def __init__(self, embedding_dim,dropout=0.1):
		"""
		Parameters:
		:: embedding_dim :: The dimension of the embeddings
		"""
		super(MLPTransformation, self).__init__()
		self.l1 = nn.Linear(embedding_dim,embedding_dim*2)
		self.l2 = nn.Linear(embedding_dim*2,embedding_dim)
		self.l = nn.Linear(embedding_dim,embedding_dim)
		# self.dropout = dropout
		self.drop = nn.Dropout(p=dropout)
		self.th = nn.Tanh()

	def forward(self, inp):
		"""
		Parameters:
		:: inp :: input embedding to be transformed
		Return:
		:: output :: transformed embedding
		"""
		output = self.l2(self.drop(self.th(self.l1(inp))))
		# output = self.l(inp)
		return output

	def remove_dropout():
		self.drop = nn.Dropout(p=0.0)

def train(transformation, inp, out, optimizer, epochs=1, save_every = 100):
	criterion = nn.MSELoss()
	optimizer.zero_grad()
	for ep in range(epochs):
		tot_loss = 0
		overall_loss = 0
		print "epoch :", ep
		for i in range(len(inp)):
			# print inp
			pout = transformation(inp[i].view(-1,len(inp[i])))
			# print pout.clone()
			# break
			loss = criterion(pout,out[i])
			tot_loss+=loss.data[0]
			loss.backward()
			if (i+1)%100 == 0:
				optimizer.step()
				optimizer.zero_grad()
				print "loss - ",i/100,":",tot_loss/100#,overall_loss
				overall_loss += tot_loss
				tot_loss = 0
				torch.save(transformation,os.getcwd()+"/transformation")
		overall_loss+=tot_loss
		print "loss - Overall:",overall_loss/len(inp)
	torch.save(transformation,os.getcwd()+"/transformation")

def transform_data(inp):
	"""
	inp: Single instance vector
	"""
	mod = {}
	if os.path.isfile(os.getcwd()+"/transformation"):
		trans = torch.load(os.getcwd()+"/transformation")
	else:
		print "No transformation file found. Please train the model first"
		return 0
	for i in (inp.wv).vocab:
		mod[i] = trans(ag.Variable(torch.FloatTensor(inp.wv[i]).view(-1,len(inp.wv[i]))))
	torch.save(mod,os.getcwd()+"/Mod_model_GENIA")

# def transform_data(inp):
# 	pass


lr = 0.001 #learning rate
model_WSJ = gensim.models.Word2Vec.load("model_WSJ")
model_GENIA = gensim.models.Word2Vec.load("model_GENIA")
vset1 = set((model_WSJ.wv).vocab)
vset2 = set((model_GENIA.wv).vocab)
vset = vset1.intersection(vset2)
l = len(vset)
print l
xi = ag.Variable(torch.FloatTensor(l,300))
yi = ag.Variable(torch.FloatTensor(l,300))
i = 0
for item in vset:
	xi[i] = torch.FloatTensor(model_GENIA.wv[item])
	yi[i] = torch.FloatTensor(model_WSJ.wv[item])
	i+=1
if os.path.isfile(os.getcwd()+"/transformation"):
	trans = torch.load(os.getcwd()+"/transformation")
else:
	trans = MLPTransformation(300)
parameters = filter(lambda p: p.requires_grad, trans.parameters())
optimizer = optim.Adam(parameters, lr=lr)
train(trans,xi,yi,optimizer,epochs=10)
transform_data(model_GENIA)

# for i in (model_WSJ.wv).vocab:
# 	if i in (model_GENIA.wv).vocab:
