import hmm_tags as HT

def createModel(N = 2):
    h = HT.TOE_HMM(N=N, maxIters = 200)
    h.loadBrownSymsSeq(100000)
    print(h.histo())
    h.initHMM()
    return h

models = [createModel(n) for n in range(2,7)]
bMats = {}

for m in models:
    print("Training model for N = %d" % m._N)
    m.trainHMM()
    m.printHMM()
    bMats[m._N] = m._hmm.emissionprob_

    print("Saving model for N = %d" % m._N)
    m.persistHMM("brown_N%d.hmm" % m._N)