from nltk.corpus import brown
import numpy as np
from hmmlearn.hmm import GMMHMM, GaussianHMM, MultinomialHMM
import nltk
import random

COMPONENTS = 256 # 0...255 for each binary ascii utf-8 value
DOT_CODE = 6
SPACE_CODE = 27

WORD_LIMIT = 100000 


class TOE_HMM_CHARS:

    def __init__(self, N=2, maxIters = 200):
        self._N = N
        self._M = COMPONENTS
        self._maxIters = maxIters
        self._syms = []

    def loadBrownSymsSeq(self, T):
        taggedWordsIter = brown.tagged_words()
        retIdx = 0
        symSequence = []
        for wrd, tag in taggedWordsIter:
            if wrd:
                for c in wrd:
                    val = ord(c)
                    symSequence.append(val)
                    retIdx += 1
            if retIdx >= T:
                break

        self._syms = symSequence
        self._syms = np.concatenate((self._syms, np.arange(256))).tolist()
        return  symSequence

    def textSeqToSymSeq(self, txtSeqArr):
        symSequence  =[]
        for wrd in txtSeqArr:
            if wrd:
                for c in wrd:
                    val = ord(c)
                    symSequence.append(val)
        return symSequence

    def initHMM(self):

        self._hmm = MultinomialHMM(n_components=self._N, n_iter=self._maxIters, 
            verbose=True, params='ste', init_params='ste')
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
        symSequence = [random.randint(0, 255) for idx in range(length)]
        return symSequence

    def pickOrderedSeq(self, length = 100):
        symSequence = []
        taggedWordsIter = brown.tagged_words()
        maxIdx = len(taggedWordsIter)
        maxMinIdx = maxIdx - length
        minIdx = random.randint(0, maxMinIdx)
        count = 0
        idx = minIdx
        while True:
            (wrd, tag) = taggedWordsIter[idx]
            if wrd:
                for c in wrd:
                    val = ord(c)
                    symSequence.append(val)
                    count += 1
            idx += 1
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
        retHisto = dict((x, self._syms.count(x)) for x in range(256))
        return retHisto

def main():

    model = TOE_HMM_CHARS(N = 2)
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

# from hmm_chars import TOE_HMM_CHARS as hc