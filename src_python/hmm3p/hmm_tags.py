from nltk.corpus import brown
import numpy as np
from hmmlearn.hmm import GMMHMM, GaussianHMM, MultinomialHMM
import nltk

COMPONENTS = 6
DOT_CODE = 6
SPACE_CODE = 27

WORD_LIMIT = 100000 
WHITE_LIST = ['NN', 'JJ', 'IN', 'VB', 'RB', '.']


class TOE_HMM:

    def __init__(self, N=2, maxIters = 200):
        self._N = N
        self._M = len(WHITE_LIST)
        self._pi = self.randProbMat(1,N)[0]
        self._A = self.equiProbMat(N, N)
        self._B = self.equiProbMat(N, self._M)
        self._maxIters = maxIters
        self._syms = []

    def randProbMat(self, M, N):
        ret = np.random.rand(M,N)
        ret = ret/ret.sum(axis=1)[:,None]
        return ret

    def equiProbMat(self, M, N):
        ret = np.ones((M,N), dtype=float)
        ret = ret/ret.sum(axis=1)[:,None]
        return ret

    def loadBrownSymsSeq(self, T):
        taggedWordsIter = brown.tagged_words()
        retIdx = 0
        iterIdx = 0
        symSequence = []
        for wrd, tag in taggedWordsIter:
            if retIdx >= T:
                break
            if tag in WHITE_LIST:
                val = WHITE_LIST.index(tag)
                symSequence.append(val)
                retIdx += 1
            iterIdx += 1

        self._syms = symSequence
        return  symSequence

    def textSeqToSymSeq(self, txtSeqArr):
        tags = nltk.pos_tag(txtSeqArr) #PerceptronTagger
        tags = [t[1] for t in tags]
        tags = [WHITE_LIST.index(t) for t in tags if t in WHITE_LIST]
        return tags

    def initHMM(self):
        # self._hmm = MultinomialHMM(n_components=self._N, startprob_prior=None, transmat_prior=None, 
        #     algorithm='viterbi', random_state=None, n_iter=self._maxIters, tol=0.01, 
        #     verbose=True, params='ste', init_params='ste')

        self._hmm = MultinomialHMM(n_components=self._N, n_iter=self._maxIters, 
            verbose=True, params='ste', init_params='ste')
        # self._hmm.emissionprob_ = self._B
        # n_features  (int) Number of possible symbols emitted by the model (in the samples).
        # monitor_    (ConvergenceMonitor) Monitor object used to check the convergence of EM.
        # transmat_   (array, shape (n_components, n_components)) Matrix of transition probabilities between states.
        # startprob_  (array, shape (n_components, )) Initial state occupation distribution.
        # emissionprob_   (array, shape (n_components, n_features)) Probability of emitting a given symbol when in each state.

    def trainHMM(self):
        self._hmm.fit(np.array(self._syms).reshape(-1, 1))

    def testTxt(self, txtSeqArr):
        testSymsArr = self.textSeqToSymSeq(txtSeqArr)
        score = self._hmm.score(testSymsArr)
        return score

    def testSyms(self, symsArr):
        # testSymsArr = self.textSeqToSymSeq(txtSeqArr)
        score = self._hmm.score(symsArr)
        return score

    def persistHMM(self, filename):
        import pickle
        s = pickle.dumps(self)
        with open(filename, 'w') as f:
            f.write(s)

    @staticmethod
    def loadHMM(filename):
        import pickle
        with open(filename, 'r') as f:
            s = f.read()
        model = pickle.loads(s)
        return model
    
    def pickRandomSeq(self, length = 100):
        symSequence = []
        taggedWordsIter = brown.tagged_words()
        maxIdx = len(taggedWordsIter)
        import random
        idx = 0
        while idx < length:
            wrdIdx = random.randint(0, maxIdx)
            (wrd, tag) = taggedWordsIter[wrdIdx]
            if tag in WHITE_LIST:
                val = WHITE_LIST.index(tag)
                symSequence.append(val)
                idx += 1
        return symSequence

    def pickOrderedSeq(self, length = 100):
        symSequence = []
        taggedWordsIter = brown.tagged_words()
        maxIdx = len(taggedWordsIter)
        maxMinIdx = maxIdx - length
        import random
        minIdx = random.randint(0, maxMinIdx)
        count = 0
        for idx in range(minIdx, maxIdx):
            (wrd, tag) = taggedWordsIter[idx]
            if tag in WHITE_LIST:
                val = WHITE_LIST.index(tag)
                symSequence.append(val)
            idx += 1
            count += 1
            if count >= length:
                break
        return symSequence

    def printHMM(self):
        print("A = %s" % str(self._hmm.transmat_))
        print("B = %s" % str(self._hmm.emissionprob_))
        print("PI = %s" % str(self._hmm.startprob_))
        print("Verify A = %s" % np.sum(self._hmm.transmat_, axis=1))
        print("Verify B = %s" % np.sum(self._hmm.emissionprob_, axis=1))
        print("Verify PI = %s" % np.sum(self._hmm.startprob_, axis=0))
    
    def histo(self):
        retHisto = dict((x, self._syms.count(x)) for x in range(len(WHITE_LIST)))
        retHisto = dict((WHITE_LIST[k], val) for k, val in retHisto.items())
        return retHisto

def main():

    model = TOE_HMM(N = 2)
    model.loadBrownSymsSeq(10000)
    print(model.histo())

    model.initHMM()
    model.trainHMM()
    model.printHMM()
    seq = model.pickRandomSeq()
    logscore = model.testSyms(seq)
    print("%s: %f" % ('Random', logscore))

    seq = model.pickOrderedSeq()
    logscore = model.testSyms(seq)
    print("%s: %f" % ('Ordered', logscore))

if __name__ == "__main__":
    main()

