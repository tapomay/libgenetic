'''
create pwmCss, pwmNbrng
define fitness
initPopulation as randomized
scoring

'''

import random
import threading
import time
from libgenetic import EvolutionBasic, Selections, Crossovers, Mutations, Generation, GABase
import numpy as np

class PWM:
    '''
    dataArr: 2-D array; column count must be consistent; all symbols must be from a set
    freqMat: [A,C,G,T][1-9]
    pwmMat: freqMat /= N
    pwmLaplaceMat: (pwmMat + 1) /= (N+4)
    odds = [A:0.22, C:.28, G:.28, T:.22]
    logoddsPwmLaplaceMat: (pwmLaplaceMat / odds)
    '''
    def __init__(self, dataArr, symbolSet, symbolOddsMap):
        self._symbolSet = symbolSet
        self._cols = 0
        self._symbolOddsMap = symbolOddsMap
        if len(dataArr) > 0:
            self._cols = len(dataArr[0])
        self._pwmMatrix = self._buildMatrix(dataArr, self._symbolSet, self._cols, symbolOddsMap)

    @property
    def pwmMatrix(self):
        return self._pwmMatrix

    def _buildMatrix(self, data2D, symbolSet, colsCount, symbolOddsMap):
        '''
        Returns:
        2D matrix; (symbolCount + 1) X cols
        Each row: position weights for a symbol at each col
        Each col: positions weights at a col for each symbol
        Last row: position weights for any unrecognized symbols at each col;
                  Caller may validate for sum(retArr[-1]) == 0 to check if input had unrecognized symbols and take precautionary actions
        
        E.g.: PWM of nucleotide 9mers : 
        self._buildMatrix(data2D:List<9MER:char[9]>, symbolCount=4, cols=9):

        BASE | pos0  | pos1  | pos2  | ... | pos8  |
        --------------------------------------------
        symA | [A,0] | [A,1] | [A,2] | ... | [A,8] |
        symC | [C,0] | [C,1] | [C,2] | ... | [C,8] |
        symG | [G,0] | [G,1] | [G,2] | ... | [G,8] |
        symT | [T,0] | [T,1] | [T,2] | ... | [T,8] |
        Unkn | [X,0] | [X,1] | [X,2] | ... | [X,8] |
        --------------------------------------------
        '''
        self.freqMat = self._frequencyCounts(data2D, symbolSet, colsCount)
        self.laplace_N = len(data2D)
        self.laplacePwmMat = self._laplaceCountPWM(self.freqMat, symbolSet, self.laplace_N)
        self.logoddsMat = self._logodds(self.laplacePwmMat, symbolSet, symbolOddsMap)
        return self.logoddsMat

    def _frequencyCounts(self, dataArr2D, symbolSet, colsCount):
        retArr = np.zeros((len(symbolSet), colsCount), dtype = float) #dtype float imp. for divisions

        symIndex = list(sorted(symbolSet))

        for row in dataArr2D:
            colIdx = 0
            for sym in row:
                if sym in symIndex:
                    symRowIdx = symIndex.index(sym)
                else:
                    raise Exception("Unrecognized symbol in input: %s; symSet: %s" % (sym, symbolSet))

                retArr[symRowIdx][colIdx] = retArr[symRowIdx][colIdx] + 1
                colIdx += 1
        return retArr

    def _laplaceCountPWM(self, frequencyMat, symbolSet, laplace_N):
        '''
            Args:
            frequencyMat: numpy nd array
        '''
        N = laplace_N # no. of rows
        N += len(symbolSet)
        pwmMat = frequencyMat.copy()
        pwmMat = pwmMat + 1
        pwmMat = pwmMat / N # avoids zero probs
        return pwmMat

    def _logodds(self, pwmMat, symbolSet, symbolOddsMap):
        ret = pwmMat.copy()
        symIndex = list(sorted(symbolSet))
        for idx in range(len(symIndex)):
            sym = symIndex[idx]
            symOdds = symbolOddsMap[sym]
            ret[idx] = ret[idx] / symOdds # -ve logvalue => less than symProb; pos => better than symProb; zero => exactly symProb
        ret = np.log2(ret)
        return ret

    def score(self, solution):
        if len(solution) != self._cols:
            raise Exception("Column count mismatch; Expected: %s" % str(self._cols))
        ret = 0
        symIndex = list(sorted(self._symbolSet))
        for col in range(self._cols):
            sym = solution[col]
            if sym in symIndex:
                symRowIdx = symIndex.index(sym)
            else:
                raise Exception("Unrecognized symbol in input: %s; symSet: %s" % (sym, self._symbolSet))

            symScore = self.pwmMatrix[symRowIdx][col]
            ret += symScore
        return ret

    @staticmethod
    def testSelf():
        dataArr=[
            ['C', 'T', 'G', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['C', 'A', 'A', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['A', 'A', 'G', 'G', 'T', 'A', 'T', 'A', 'T'],
            ['A', 'G', 'T', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['T', 'T', 'G', 'G', 'T', 'A', 'A', 'A', 'A'],
            ['T', 'G', 'G', 'G', 'T', 'A', 'A', 'G', 'G'],
            ['C', 'A', 'G', 'G', 'T', 'G', 'A', 'G', 'T'],
            ['A', 'G', 'G', 'G', 'T', 'A', 'A', 'T', 'G'],
            ['T', 'A', 'G', 'G', 'T', 'A', 'T', 'T', 'G'],
            ['C', 'A', 'G', 'G', 'T', 'A', 'A', 'A', 'A'],
            ['A', 'A', 'G', 'G', 'T', 'G', 'T', 'G', 'T'],
            ['A', 'A', 'G', 'G', 'T', 'A', 'A', 'G', 'A'],
            ['T', 'A', 'G', 'G', 'T', 'A', 'A', 'T', 'A'],
            ['T', 'T', 'T', 'G', 'T', 'G', 'A', 'G', 'T'],
            ['C', 'A', 'G', 'G', 'T', 'A', 'T', 'A', 'C'],
            ['T', 'C', 'T', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['G', 'A', 'G', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['A', 'A', 'G', 'G', 'T', 'A', 'A', 'A', 'G'],
            ['C', 'A', 'G', 'G', 'T', 'A', 'A', 'G', 'A'],
            ['A', 'C', 'A', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['A', 'T', 'G', 'G', 'T', 'A', 'A', 'G', 'G']
        ]
        symbolSet = set(['A', 'C', 'G', 'T'])
        symbolOddsMap = {'A':0.28, 'C': 0.22, 'G': 0.22, 'T': 0.28}
        pwm = PWM(dataArr, symbolSet, symbolOddsMap)
        print(pwm.pwmMatrix)

        testSolution = "CTGGTAAGT"
        testScore = pwm.score(testSolution)
        print(testScore)
        return pwm

class GASpliceSites:
    '''
        Wrapper for splice site 9mers and PWM
    '''
    BASES_MAP = {0:'A', 1:'C', 2:'G', 3:'T'}
    @staticmethod
    def load_data_tsv(filename):
        '''
            Accepts tab separated 9-mers
        '''
        ret = []
        with open(filename) as f:
            content = f.read()
            lines = content.split('\n')
            for line in lines:
                bases = line.split('\t')
                ret.append(bases)
        return ret

    def __init__(self, nmerArr):
        self._nmerArr = nmerArr
        self._pwm = self._computePwm(nmerArr)

    def _computePwm(self, nmerArr):
        symbolSet = set(['A', 'C', 'G', 'T'])
        symbolOddsMap = {'A':0.28, 'C': 0.22, 'G': 0.22, 'T': 0.28}
        pwm = PWM(nmerArr, symbolSet, symbolOddsMap)
        return pwm

    def isValid9merSpliceSite(self, ninemer):
        if len(ninemer) != 9:
            raise Exception("dimension mismatch")
        return ninemer[3] == 'G' and ninemer[4] == 'T'

    def fitness(self, ninemer):
        if len(ninemer) != 9:
            raise Exception("dimension mismatch")
        baseScore = self._pwm.score(ninemer)
        penalty = 0
        if not self.isValid9merSpliceSite(ninemer):
            # print("Invalid ninemer: %s" % ninemer)
            penalty = 9 * 1000 #TODO: max logodds score; log2(1 / min(symOdds))
        ret = baseScore - penalty
        return ret

    def baseFlip(self, baseValue):
        '''
        Return any random base other than given base
        '''
        randomBase = random.randint(0, 3)
        flipValue = GASpliceSites.BASES_MAP[randomBase]
        while flipValue == baseValue:
            randomBase = random.randint(0, 3)
            flipValue = GASpliceSites.BASES_MAP[randomBase]
        return flipValue

    def __str__(self):
        pass
        # return "P: %s\nW: %s\nC: %s\nS: %s" % (str(self._profits), str(self._weights), str(self._capacity), str(self._solution))

class GASpliceSitesThread(threading.Thread):
    def __init__(self, gASpliceSites, initPopulation, genCount):
        threading.Thread.__init__(self)
        self._gASpliceSites = gASpliceSites
        self._initPopulation = initPopulation
        self._genCount = genCount
        self._gaBase = None

    def run(self):
        gen0 = Generation(self._initPopulation)
        recombine = lambda population: Selections.rouletteWheel(population, self._gASpliceSites.fitness)
        mutator = lambda solution: Mutations.provided_flip(solution, flipProvider = self._gASpliceSites.baseFlip)
        evolution = EvolutionBasic(select = recombine, crossover = Crossovers.two_point, mutate = mutator)
        gaBase = GABase(evolution, gen0, self._gASpliceSites.fitness)
        gaBase.execute(maxGens=self._genCount)
        self._gaBase = gaBase

    @property
    def gaBase(self):
        return self._gaBase

def randomSpliceSitesPopulation(M, N, cardinality=4):
    randomgen = lambda: random.randint(0, cardinality - 1)
    basesMap = GASpliceSites.BASES_MAP
    ret = []
    for i in range(M):
        sol = []
        for j in range(N):
            if j == 3:
                bit = 2 #'G'
            elif j == 4:
                bit = 3 #'T'
            else:
                bit = randomgen()
            sol.append(basesMap[bit])
        ret.append(sol)
    return ret

def check_match(population, ninemerData):
    '''
    Args:
    population: candidate 9mers
    ninemerData: training 9mers
    Returns:
        percent match of population in ninemerData
    '''
    ninemerStrData = ["".join(nm) for nm in ninemerData]
    populationStrData = ["".join(nm) for nm in population]
    ret = 0
    for solution in populationStrData:
        if solution in ninemerStrData:
            ret += 1

    ret /= float(len(populationStrData))
    return ret

def find_best_gens(gaBase):
    genFitness = [gen._bestFitness for gen in gaBase._generations]
    bestFitness_all = max(genFitness)
    bestGenArr = filter(lambda g: g._bestFitness == bestFitness_all, gaBase._generations)
    return bestGenArr

def match_stat(gaBase, authssData, cssData):
    lastGen = gaBase._generations[-1]
    bestGens = find_best_gens(gaBase)
    # bestgen_scoreCss = [check_match(gen._population, cssData) for gen in bestGens]
    # bestgen_scoreCss = max(bestgen_scoreCss)
    # bestgen_scoreAuth = [check_match(gen._population, authssData) for gen in bestGens]
    # bestgen_scoreAuth = max(bestgen_scoreAuth)
    bestgen_scoreCss = -float('inf')
    bestgen_scoreAuth = -float('inf')
    bestgen_genCss = -1
    bestgen_genAuth = -1

    for gen in bestGens:
        scoreCss = check_match(gen._population, cssData)
        if scoreCss > bestgen_scoreCss:
            bestgen_scoreCss = scoreCss
            bestgen_genCss = gen._genIndex
        scoreAuth = check_match(gen._population, authssData)
        if scoreAuth > bestgen_scoreAuth:
            bestgen_scoreAuth = scoreAuth
            bestgen_genAuth = gen._genIndex

    lastgen_scoreCss = check_match(lastGen._population, cssData)
    lastgen_scoreAuth = check_match(lastGen._population, authssData)
    return (bestgen_scoreCss, bestgen_scoreAuth, bestgen_genCss, bestgen_genAuth, lastgen_scoreCss, lastgen_scoreAuth)

def main(cssFile = 'data/splicesite_data/CrypticSpliceSite.tsv', 
    authssFile = 'data/splicesite_data/EI_true_9.tsv', 
    generationSize = 10, genCount = 10):
    cssGAData = GASpliceSites.load_data_tsv(cssFile)
    authssGAData = GASpliceSites.load_data_tsv(authssFile)
    cssGASpliceSites = GASpliceSites(cssGAData)
    authGASpliceSites = GASpliceSites(authssGAData)

    M = generationSize
    N = 9 # 9-mers
    initPopulation = randomSpliceSitesPopulation(M, N)
    print(initPopulation)

    authThread = GASpliceSitesThread(authGASpliceSites, initPopulation, genCount = genCount)
    cssThread = GASpliceSitesThread(cssGASpliceSites, initPopulation, genCount = genCount)

    cssThread.start()
    cssThread.join()
    cssGABase = cssThread.gaBase

    authThread.start()
    authThread.join()
    authGABase = authThread.gaBase

    stats = []
    stats.append(['TRAINER', 'bestgen_scoreCss', 'bestgen_scoreAuth', 'bestgen_genCss', 'bestgen_genAuth', 'lastgen_scoreCss', 'lastgen_scoreAuth'])
    
    (bestgen_scoreCss, bestgen_scoreAuth, bestgen_genCss, bestgen_genAuth, lastgen_scoreCss, lastgen_scoreAuth) = \
        match_stat(cssGABase, authssGAData, cssGAData)
    print("\nCSS GAStats:")
    print("BESTGEN: cssGen_X_cssData: %s" % str(bestgen_scoreCss))
    print("BESTGEN: cssGen_X_authData: %s" % str(bestgen_scoreAuth))
    print("BESTGENIDX: cssGen_X_cssData: %s" % str(bestgen_genCss))
    print("BESTGENIDX: cssGen_X_authData: %s" % str(bestgen_genAuth))
    print("LASTGEN: cssGen_X_cssData: %s" % str(lastgen_scoreCss))
    print("LASTGEN: cssGen_X_authData: %s" % str(lastgen_scoreAuth))
    stats.append(['cssGABase', bestgen_scoreCss, bestgen_scoreAuth, bestgen_genCss, bestgen_genAuth, lastgen_scoreCss, lastgen_scoreAuth])

    (bestgen_scoreCss, bestgen_scoreAuth, bestgen_genCss, bestgen_genAuth, lastgen_scoreCss, lastgen_scoreAuth) = \
        match_stat(authGABase, authssGAData, cssGAData)
    print("\nAUTH GAStats:")
    print("BESTGEN: authGen_X_cssData: %s" % str(bestgen_scoreCss))
    print("BESTGEN: authGen_X_authData: %s" % str(bestgen_scoreAuth))
    print("BESTGENIDX: authGen_X_cssData: %s" % str(bestgen_genCss))
    print("BESTGENIDX: authGen_X_authData: %s" % str(bestgen_genAuth))
    print("LASTGEN: authGen_X_cssData: %s" % str(lastgen_scoreCss))
    print("LASTGEN: authGen_X_authData: %s" % str(lastgen_scoreAuth))
    stats.append(['authGABase', bestgen_scoreCss, bestgen_scoreAuth, bestgen_genCss, bestgen_genAuth, lastgen_scoreCss, lastgen_scoreAuth])

    from tabulate import tabulate
    print("\nRESULTS:")
    print(tabulate(stats, headers='firstrow'))
    return stats

if __name__ == '__main__':
    main(generationSize = 10, genCount = 50)