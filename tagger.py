import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	# Edit here
	S = len(tags)
	state_dict = dict()

	for i in range(S):
		state_dict[tags[i]] = i

	sentences = len(train_data)
	pi = [0.0] * S
	for sentence in range(sentences):
		ind = state_dict[train_data[sentence].tags[0]]
		pi[ind] = pi[ind] + 1

	for i in range(S):
		pi[i] = pi[i] / sentences

	obs_dict = {}
	ind = 0
	for sentence in range(sentences):
		sen = train_data[sentence].words
		for word in sen:
			if word not in obs_dict.keys():
				obs_dict[word] = ind
				ind = ind + 1

	A = np.zeros([S, S])
	start = [0.0] * S
	for sentence in range(sentences):
		sen = train_data[sentence].tags
		for i in range(len(sen) - 1):
			s = sen[i]
			start[state_dict[s]] += 1
			sp = sen[i + 1]
			A[state_dict[s]][state_dict[sp]] += 1

	for i in range(S):
		if start[i] != 0:
			A[i] = [x / start[i] for x in A[i]]

	B = np.zeros([S, ind])
	for sentence in range(sentences):
		start[state_dict[train_data[sentence].tags[-1]]] += 1
		for i in range(len(train_data[sentence].tags)):
			s = train_data[sentence].tags[i]
			o = train_data[sentence].words[i]
			B[state_dict[s]][obs_dict[o]] += 1

	for i in range(S):
		if start[i] != 0:
			B[i] = [x / start[i] for x in B[i]]

	model = HMM(pi, A, B, obs_dict, state_dict)
	###################################################
	return model

# TODO:
def sentence_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	# Edit here
	num_tags = len(tags)
	for i, d in enumerate(test_data):
		for word in d.words:
			if word not in model.obs_dict:
				model.obs_dict[word] = len(model.obs_dict)
				e_col = np.full((num_tags, 1), 1 ** -6)
				model.B = np.hstack((model.B, e_col))
		loc = model.viterbi(d.words)
		tagging.append(loc)
	###################################################
	return tagging
