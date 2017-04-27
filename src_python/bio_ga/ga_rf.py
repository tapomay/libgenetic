import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
import pandas as pd
from ga_css_5prime import MatchUtils
import tempfile
from sklearn import preprocessing

np.random.seed(0)

CSS_5PRIME_FILE = 'data/dbass-prats/CrypticSpliceSite.tsv'
AUTH_5PRIME_FILE = 'data/hs3d/Exon-Intron_5prime/EI_true_9.tsv'
NEIGH_5PRIME_FILE = 'data/dbass-prats/NeighboringSpliceSite.tsv'
CSS_3PRIME_FILE = 'data/dbass/css_3prime.tsv'
AUTH_3PRIME_FILE = 'data/hs3d/Intron-Exon_3prime/authss_3prime.tsv'

def convert():
	le = preprocessing.LabelEncoder()

def loadTestData():
	A5 = pd.read_csv(AUTH_5PRIME_FILE, sep='\t', header = None)
	C5 = pd.read_csv(CSS_5PRIME_FILE, sep='\t', header = None)
	N5 = pd.read_csv(NEIGH_5PRIME_FILE, sep='\t', header = None)

	A5_label = pd.DataFrame(np.repeat(0, A5.shape[0]))
	C5_label = pd.DataFrame(np.repeat(1, C5.shape[0]))
	N5_label = pd.DataFrame(np.repeat(2, N5.shape[0]))

	# A5 = pd.concat([A5_data, A5_label], axis = 1)
	# C5 = pd.concat([A5_data, A5_label], axis = 1)
	# N5 = pd.concat([A5_data, A5_label], axis = 1)

	X_pos_test = pd.concat([A5, C5], axis=0)
	X_neg_test = N5

	X_pos_test = X_pos_test.as_matrix()
	X_neg_test = X_neg_test.as_matrix()

	return (X_pos_test, X_neg_test)

def extract_best_byfitness(perfMap, dataname): #neighbor/auth/css
	
	bestGABase = None
	bestFitness = -float('inf')
	(retselect, retxover, retdataname, retM, retN, retbestGABase, retbestFitness) = (None, None, None, None, None, None, None)
	for (select, xover), perf1 in perfMap.items():
		select = select.upper().replace('_', '-')
		xover = xover.upper().replace('CROSSOVER', '').replace('_', '')
		dataPerfs = dict()
		for (M,N,dataname), perf2 in perf1.items():
			k = (M,N)
			if dataname not in dataPerfs:
				dataPerfs[dataname] = dict()
			dataPerfs[dataname][k] = perf2


		dimPerfMap = dataPerfs[dataname]
		for M,N in sorted(dimPerfMap):
			dimPerf = dimPerfMap[(M,N)]
			bestGens = MatchUtils.find_best_gens(dimPerf.bestGABase)
	        fitness = None
	        if bestGens:
	            fitness = bestGens[0]._bestFitness
	            if fitness > bestFitness:
	            	bestFitness = fitness
	            	bestGABase = dimPerf.bestGABase
	            	(retselect, retxover, retdataname, retM, retN, retbestGABase, retbestFitness) = (select, xover, dataname, M, N, bestGABase, bestFitness)
	return (retselect, retxover, retdataname, retM, retN, retbestGABase, retbestFitness)

def extractPopulation(gaBase):
	ret = []
	genIndices = set()
	for gen in gaBase._generations:
		if gen._genIndex not in genIndices:
			genIndices.add(gen._genIndex)
			ret.append(gen._bestSolution)
	return ret

def extractTrainingData(perfMap):
	datanames = {'auth':0, 'css':0, 'neighbor':1}
	X_trains = []
	y_trains = []
	for dataname, label in datanames.items():
		(select, xover, dataname, M, N, bestGABase, bestFitness) = extract_best_byfitness(perfMap, dataname)
		population = extractPopulation(bestGABase)
		poparr = ["\t".join(pop) for pop in population if pop]
		popstr = "\n".join(poparr)
		tmpFile = tempfile.mkstemp()
		with open(tmpFile[1], 'w') as f:
			f.write(popstr)
		pdpop = pd.read_csv(tmpFile[1], sep='\t', header = None)
		poplen = pdpop.shape[0]
		labels = pd.DataFrame(np.repeat(label, poplen))
		X_trains.append(pdpop)
		y_trains.append(labels)

	retX = pd.concat(X_trains, axis = 0)
	retY = pd.concat(y_trains)

	retX = retX.as_matrix()
	retY = retY.as_matrix()
	return (retX, retY)

def encode(le, data):
	ret = [le.transform(d) for d in data]
	return ret

# def execute_score(perfMap):
# 	(X_pos_test, X_neg_test) = loadTestData()
# 	(X_train, y_train) = extractTrainingData(perfMap)
# 	le = preprocessing.LabelEncoder()
# 	le.fit(['A', 'C', 'G', 'T'])
# 	X_test = encode(le, X_test)
# 	# y_test = encode(le, y_test)
# 	X_train = encode(le, X_train)
# 	# y_train = encode(le, y_train)

# 	clf = RandomForestClassifier(n_estimators=25)
# 	clf.fit(X_train, y_train)
# 	y_pos = clf.predict_proba(X_pos_test)
# 	y_neg = clf.predict_proba(X_neg_test)
# 	# y_predict = clf.predict(X_test)
# 	# score = log_loss(y_test, clf_probs)
# 	# print(score)
# 	return (y_predict, y_test, score)

def execute(perfMap):
	(X_pos_test, X_neg_test) = loadTestData()
	(X_train, y_train) = extractTrainingData(perfMap)
	le = preprocessing.LabelEncoder()
	le.fit(['A', 'C', 'G', 'T'])
	X_train = encode(le, X_train)
	X_pos_test = encode(le, X_pos_test)
	X_neg_test = encode(le, X_neg_test)

	clf = RandomForestClassifier(n_estimators=25)
	clf.fit(X_train, y_train)
	y_pos = clf.predict_proba(X_pos_test)
	y_neg = clf.predict_proba(X_neg_test)
	# y_predict = clf.predict(X_test)
	# score = log_loss(y_test, clf_probs)
	# print(score)
	return (y_pos, y_neg)

def script():
	(y_pos, y_neg) = garf.execute(perfMap)
	
	y_pos_str = [str(p) for p in y_pos]
	pos_str = "\n".join(y_pos_str)

	y_neg_str = [str(p) for p in y_neg]
	neg_str = "\n".join(y_neg_str)

	with open('negatives','w') as f:
	    f.write(neg_str)
	
	with open('positives','w') as f:
        f.write(pos_str)
