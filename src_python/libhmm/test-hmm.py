import unittest
from hmm import *

logger = logging.getLogger("HMM_MAIN")
logger.setLevel(logging.DEBUG)
logger = logging.getLogger("HMM_TEST")
logger.setLevel(logging.DEBUG)
FORMAT = '%(name) %(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)

class TemperatureCorpus:
    def __init__(self, seqlen = 4):
        self.seqlen = seqlen

    def obsSeq(self):
        ret = ['0', '1', '0', '2']
        self._count = 3 # S/M/L
        return ret

    @property
    def count(self):
        if(self._count):
            return self._count
        else:
            raise Exception("No generated sequence found. Call obsSeq first.")

class TemperatureLambdaBuilder:
    @property
    def A(self):
        ret = np.array([[0.7,0.3], [0.4,0.6]])
        return ret

    @property
    def B(self):
        ret = np.array([[0.1,0.4,0.5], [0.7,0.2,0.1]])
        return ret

    @property
    def pi(self):
        ret = np.array([0.6, 0.4])
        return ret

class TestHmmMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("SETUP:")
        corpus = TemperatureCorpus(4)
        cls.obsSeq = corpus.obsSeq()
        M = corpus.count
        print("Obs seq length: T=%d; Obs.syms.count=%d" % (len(cls.obsSeq), M))
        lambdaBuilder = TemperatureLambdaBuilder()

        minIters=-1
        maxIters=-1
        threshold=-1
        cls.hmm = HMM(minIters, maxIters, threshold, lambdaBuilder)

    def notest_alpha_pass(self):
        global logger
        logger.info("PI: %s" % str(self.hmm.pi))
        logger.info("A: %s" % str(self.hmm.A))
        logger.info("B: %s" % str(self.hmm.B))

        (alpha, c) = self.hmm.alphaPass(self.obsSeq)
        logger.info("Final - alpha: \n%s" % str(alpha))
        logger.info("c: %s" % str(c))

    def notest_beta_pass(self):
        global logger
        testC = [1, 1/2, 1/3, 1/4]
        beta = self.hmm.betaPass(testC, self.obsSeq)
        logger.info("Final Beta: \n%s" % str(beta))

    def test_gamma_pass(self):
        global logger
        
        testA = [[2,3],[5,7]]
        testB = [[11,13,17],[19,23,29]]

        testAlpha = np.array([[31,37], [41,43], [47,53], [59,61]])
        testBeta = np.array([[67,71], [73,79], [83,89], [97,101]])

        self.hmm.A = np.array(testA)
        self.hmm.B = np.array(testB)

        (gamma, digamma) = self.hmm.gammaPass(testAlpha, testBeta, self.obsSeq)
        
        logger.info("Final Gamma: \n%s" % str(gamma))
        logger.info("Final DiGamma: \n%s" % str(digamma))

if __name__ == '__main__':
    unittest.main()