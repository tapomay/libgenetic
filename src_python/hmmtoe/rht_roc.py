pat = 'rach297/finals_data'

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(pat) if isfile(join(pat, f)) and f.endswith('.enc')]

print(onlyfiles)

LIMIT = 5000

def model_scores(model):
    for fname in onlyfiles:
        fpath = join(pat, fname)
        with open(fpath) as f:
            fstr = f.read()
            fstr = fstr.replace('.', ' . ')
            farr = fstr.split()
            if len(farr) > LIMIT:
                farr = farr[0:LIMIT]

            score = model.testTxt(farr)
            score /= len(farr)
            print("fname:%s" % fname)
            print("len:%d score:%f" %(len(farr), score))
            print("positive:%f" %(score))

            rseq = model.pickRandomSeq(len(farr))
            scoreRand = model.testSyms(farr)
            scoreRand /= len(farr)
            print("negative:%f" %(scoreRand))

            with open("positives_%d" % model._N, 'a') as f:
                f.write("%f\n" % score)

            with open("negatives_%d" % model._N, 'a') as f:
                f.write("%f\n" % scoreRand)

def all_scores(models):
    for m in models:
        model_scores(m)

model_files = ['brown_N2.hmm', 'brown_N3.hmm', 'brown_N4.hmm', 'brown_N5.hmm', 'brown_N6.hmm']

import pickle

models = []
for model_f in model_files:
    with open(model_f) as f:
        s = f.read()
        m = pickle.loads(s)
        models.append(m)

all_scores(models)