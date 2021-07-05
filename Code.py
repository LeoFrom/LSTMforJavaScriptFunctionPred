import pickle
import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from collections import defaultdict
import itertools


#Preprocess of datas -> unpickle the file containing all datas
# 					 -> Data visualisation
# 					 -> Create a train and test set to train the model / then evaluate accuracy of model (ratio : 80-20)
with open('anulap.pkl', 'rb') as f:
	my_depickler = pickle.Unpickler(f)
	datas = my_depickler.load()

print(datas)
# df = pd.DataFrame(datas, columns= ['function', 'types','project'])
# row = df.values.tolist()
# for item in row:
# 	print(item)
# 	print()
# 	input()

#dictionnary of each parameter with the text and his class
#get the vocabulary from the parameters of every functions
#transform data to get gold array from labels array
#representation of each parameter by it's word embedding found using pytorch .nn. embeddings
# apply model and get prediction and rectify if prediction wrong which model ?

#Get the vocabulary and the parameter to match to each type
vocabulary = set()
words_to_predict = []
X_set = []
recap = defaultdict(str)
voc = datas['function'].tolist()
for function in voc :
	#Get the parameters of the function
	splited_function = function.split("\n")
	parameter_of_actual_function = []
	start,end = 0,0
	for character in splited_function[0]:
		if character == "(":
			start = splited_function[0].find('(')
		if character == ")":
			end = splited_function[0].find(')')
		if start != 0 and end != 0:
			parameters = splited_function[0][start+1:end]
			cleaned_param = re.sub(r'/.+?/', '', parameters)
			cleaned_param_2 = cleaned_param.split(":")
			cleaned_param_2 = str(cleaned_param_2[0]) #take care of the already assigned type
			#print(cleaned_param_2)
			parameter_of_actual_function.append(cleaned_param_2)
			break
	joined ="".join(parameter_of_actual_function).replace(" ","").split(",")
	words_to_predict.extend(joined)

	#Get the totality of the vocabulary
	splited_function_2 = re.split(r'[.\s(),]\s*', function)
	X_set.append(splited_function_2)
	for element in splited_function_2:
		vocabulary.add(element)

# print(words_to_predict)
# print(X_set)

Y_set = []
types = datas['types'].tolist()
for set_type in types:
	Y_types = set_type.split(",")
	for one_elem in Y_types:
		Y_set.append(one_elem)
# print(Y_set)

word2idx = {word: ind for ind, word in enumerate(vocabulary)}
# gold_values = zip(words_to_predict,Y_set)
# for one_gold in gold_values:
# 	print(one_gold)
# 	input()

labels = []
for items in types:
	all_type = [item for item in items.split(',')]
	for final_type in all_type:
		labels.append(final_type)
labels = set(labels)
labels2idx = {label: ind for ind, label in enumerate(labels)}
#print(labels2idx)

X_encoded_set = []
for function in words_to_predict:
	encoded_function = [word2idx[function]]
	X_encoded_set.append(encoded_function)

# X_encoded_set = list(itertools.chain.from_iterable(X_encoded_set))
# input()

Y_encoded_set = []
for types in Y_set:
	encoded_type = [labels2idx[types]]
	Y_encoded_set.append(encoded_type)

# print(Y_encoded_set)
# input()


reconstructed_encod_datas = list(zip(X_encoded_set,Y_encoded_set))
#print(reconstructed_encod_datas)
train, test = train_test_split(reconstructed_encod_datas, random_state=42, test_size=0.20, shuffle=True)
#print(train)
# print(len(train),len(test))
#split the datas from the types
#print(train)

# input()
# res_train = [[ i for i, j in train ], 
# 	   [ j for i, j in train ]] 
# X_train , Y_train = res_train
# print(res_train)


vocab_size = len(vocabulary)
emb_dim = 5
hidden_dim = 5

# emb_layer = nn.Embedding(vocab_size, emb_dim)
# emb_datas = torch.LongTensor(X_encoded_set)
# embeddings = emb_layer(emb_datas)
# input()

class LSTM(torch.nn.Module) :
	def __init__(self, vocab_size, embedding_dim, hidden_dim) :
		super().__init__()
		self.hidden_dim    = hidden_dim
		self.embedding_dim = embedding_dim

		self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=False)
		self.out = nn.Linear(hidden_dim, len(labels2idx))
		self.dropout = nn.Dropout(0.2)
		
	def forward(self, X):
		emb_datas = torch.LongTensor(X)
		embedded = self.embeddings(emb_datas)
		embedded = self.dropout(embedded)
		lstm_out, (hidden, cell) = self.lstm(embedded.unsqueeze(0))
		return self.out(hidden)

	def train(self,train,epochs,lr=0.001):
		train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle =True)

		loss_func  = nn.CrossEntropyLoss() 
		optimizer  = optim.Adam(self.parameters(), lr=lr)
		#For shallow networks but improve slower 
		#optimizer  = optim.SGD(self.parameters(), lr=0.01)

		for e in range(epochs):
			train_logloss = 0.0
			correct = 0
			for x,y in train_loader:
				self.zero_grad()
				pred            = self.forward(x)
				predicted = torch.argmax(pred)
				loss            = loss_func(pred.squeeze(0),y[0])
				loss.backward()
				optimizer.step()
				train_logloss += loss.item()
				if (predicted == y[0]):
					correct += 1 

			print("Epoch:", e, "train_loss:" , train_logloss/len(train), "accuracy:" , correct/len(train))

lstm = LSTM(vocab_size,emb_dim,hidden_dim)
lstm.train(train,20)