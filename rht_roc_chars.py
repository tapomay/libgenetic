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
            if len(fstr) > LIMIT:
                fstr = fstr[0:LIMIT]

            score = model.testTxt(fstr)
            score /= len(fstr)
            print("fname:%s" % fname)
            print("len:%d score:%f" %(len(fstr), score))
            print("positive:%f" %(score))

            rseq = model.pickRandomSeq(len(fstr))
            scoreRand = model.testSyms(rseq)
            scoreRand /= len(fstr)
            print("negative:%f" %(scoreRand))

            with open("positives_chars_%d" % model._N, 'a') as f:
                f.write("%f\n" % score)

            with open("negatives_chars_%d" % model._N, 'a') as f:
                f.write("%f\n" % scoreRand)

def all_scores(models):
    for m in models:
        model_scores(m)

model_files = ['brown_chars_N2.hmm', 'brown_chars_N3.hmm', 'brown_chars_N4.hmm', 'brown_chars_N5.hmm', 'brown_chars_N6.hmm']

import pickle

models = []
for model_f in model_files:
    with open(model_f) as f:
        s = f.read()
        m = pickle.loads(s)
        models.append(m)

all_scores(models)