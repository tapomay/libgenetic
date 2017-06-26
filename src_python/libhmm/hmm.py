import numpy as np

'''
Homework 1
Problem 10
Problem type: 3 - parameter estimation
'''

class HMM:

    """
        Steps:
            Compute alpha[0] : 1XN
            Scale alpha[0]
            Compute alpha[1:T-1] : TXN
            Scale alpha[1:T-1]

            Compute beta[T-1] : 1XN
            Compute scaled beta[T-2:0] : TXN

            for each t:
                Compute denom : 1X1
                Compute digamma : NXN
                Compute gamma : 1XN

            digamma: TXNXN
            gamma: TXN

            Re-estimate pi: 1XN
            Re-estimate A: NXN
            Re-estimate B: MXN

            Compute logProb
            Iteration logic
            End
    """

    def __init__(self, minIters, threshold, lambdaProvider):
        self.minIters = minIters
        self.threshold = threshold
        self.iters = 0
        self.oldLogProb = -np.inf

        self.A = lambdaProvider.A
        self.B = lambdaProvider.B
        self.pi = lambdaProvider.pi


    '''
    Assuming obsSeq[] has T symbols 0...M-1
    '''
    def alphaPass(self, obsSeq):
        A = self.A
        B = self.B
        pi = self.pi
        T = len(obsSeq)
        N = np.shape(A)[0]

        o0 = int(obsSeq[0])
        alpha = np.empty((T, N))
        c = np.empty(T)
        alpha[0] = np.multiply(pi, np.transpose(B[:, o0])) #1XN
        c[0] = alpha[0].sum(axis=0)
        c[0] = 1 / c[0]
        # print(c[0])
        alpha[0] = alpha[0] * c[0]

        # alpha = np.transpose(alpha) #column matrix
        for t in range(1, len(obsSeq)):
            o = int(obsSeq[t])
            alphaTmp = A * alpha[t-1][:, None]
            alpha[t] = alphaTmp.sum(axis = 0)
            alpha[t] = np.multiply(alpha[t], B[:, o])
            c[t] = alpha[t].sum(axis = 0)
            c[t] = 1 / c[t]
            alpha[t] = alpha[t] * c[t]

        return (alpha, c)

    def betaPass(self, alpha, c, obsSeq):
        A = self.A
        B = self.B
        pi = self.pi
        T = len(obsSeq)
        N = np.shape(A)[0]
        beta = np.empty((T, N))
        beta[T-1] = np.ones(N) * c[T-1]

        t = T-2
        while t>=0:
            o = int(obsSeq[t+1])
            bCol = np.transpose(B[:,o])
            betaRow = beta[t+1]
            betaTmp = A * bCol[:,None] * betaRow[:,None]
            beta[t] = betaTmp.sum(axis=0)
            beta[t] = beta[t] * c[t]
            t = t - 1
        return beta


    def gammaPass(self, alpha, beta, obsSeq):
        A = self.A
        B = self.B
        pi = self.pi
        T = len(obsSeq)
        N = np.shape(A)[0]
        denom = np.empty(T)
        gamma = np.empty([T,N])
        digamma = np.empty([T,N,N])

        for t in range(0, T-2):
            o = int(obsSeq[t+1])
            bCol = np.transpose(B[:,o])
            betaRow = beta[t+1]
            alphaRow = alpha[t]
            denomTmp = A * alphaRow[:,None] * bCol[:,None] * betaRow[:,None]
            denom[t] = denomTmp.sum(axis=0).sum()
            digamma[t] = denomTmp / denom[t]
            gamma[t] = digamma[t].sum(axis=1)

        denom = alpha[T-1].sum()
        gamma[T-1] = alpha[T-1] / denom

        return(gamma, digamma)

    def reestimate(self, gamma, digamma, obsSeq):
        newPi = gamma[0]
        B = self.B
        M = np.shape(B)[1]
        N = np.shape(B)[0]
        T = len(obsSeq)

        digammaSumAcrossT = digamma[0:T-2].sum(axis=0)
        gammaSumAcrossT = gamma[0:T-2].sum(axis=0)
        newA = digammaSumAcrossT / gammaSumAcrossT

        newB = np.empty([N,M])
        for i in range(0, N-1):
            for m in range(0,M):
                numer = 0
                denom = 0
                for t in range(0, T-1):
                    o = int(obsSeq[t])
                    denom = denom + gamma[t,i]
                    if m == o:
                        numer = numer + gamma[t,i]
                newB[i,m] = numer/denom

        return (newPi, newA, newB)

    def logProb(self, c):
        logC = np.log(c)
        logProb = logC.sum()
        return -logProb

    def checkPerf(self, logProb):
        self.iters = self.iters + 1
        delta = abs(logProb - self.oldLogProb)
        if(self.iters < self.minIters or delta > self.threshold):
            print("Iter: %s, minIters: %s, delta:%f, thresh:%f" % (self.iters, self.minIters, delta, self.threshold) )
            self.oldLogProb = logProb
            return True
        else:
            return False

    def executeType3(self, obsSeq):

        doIterate = True
        # print(obsSeq)
        while(doIterate == True):
            (alpha, c) = self.alphaPass(obsSeq)

            # print("ALPHA: " + str(np.shape(alpha)))
            # print("C" + str(np.shape(c)))
            # print("ALPHA: " + str(alpha))
            # print("C" + str(c))

            beta = self.betaPass(alpha, c, obsSeq)
            # print("BETA: " + str(np.shape(beta)))
            # print("BETA: " + str(beta))

            (gamma, digamma) = self.gammaPass(alpha, beta, obsSeq)
            # print("GAMMA" + str(np.shape(gamma)))
            # print("DIGAMMA" + str(np.shape(digamma)))

            (newPi, newA, newB) = self.reestimate(gamma, digamma, obsSeq)
            # print("newPi: " + str(np.shape(newPi)))
            # print("newA: " + str(np.shape(newA)))
            # print("newB: " + str(np.shape(newB)))
            logProb = self.logProb(c)
            doIterate = self.checkPerf(logProb)
            self.A = newA
            self.B = newB
            self.pi = newPi
            # print(newA)
            print("Iteration#%d: logProb=%f" % (self.iters, logProb))
            # break

class BrownCorpus:

    def convert(self, w):
        ret = []
        for c in w:
            if c == ' ':
                ret.append('26')
            else:
                ret.append(str(ord(c) - ord('a')))
        return ret

    def revconvert(self, arr):
        ret = []
        for a in arr:
            c = chr(a + ord('a'))
            ret.append(c)
        return ret

    def obsSeq(self):
        from nltk.corpus import brown
        wl = brown.words()[0:2000]
        ret = []
        for w in wl:
            w = w.lower()
            if(w.isalpha()):
                ret = ret + self.convert(w)
                ret = ret + self.convert(" ")

        return ret

class LambdaBuilder:

    def __init__(self, N, M):
        self.N = N
        self.M = M

    @property
    def A(self):
        ret = np.ones([self.N, self.N])
        ret = ret / self.N
        # ret = [[0.47, 0.53],[0.51, 0.49]]
        # print("A= " + str(ret))
        return ret

    @property
    def B(self):
        ret = np.ones([self.N, self.M])
        ret = ret / self.M
        # print("B= " + str(ret))
        return ret

    @property
    def pi(self):
        ret = np.ones([1, self.N])
        ret = ret / self.N
        # pi = [0.51, 0.49]
        # print("PI= " + str(ret))
        return ret

def main():
    obsSeq = BrownCorpus().obsSeq()
    lambdaBuilder = LambdaBuilder(2, 27)

    hmm = HMM(100, 10000, lambdaBuilder)
    hmm.executeType3(obsSeq)
    print(hmm.A)
    print(hmm.B)

if __name__ == "__main__":
    main()