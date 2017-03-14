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

def main(GEN_COUNT = 20):
    cssFile = 'data/splicesite_data/CrypticSpliceSite.tsv'
    authssFile = 'data/splicesite_data/EI_true_9.tsv'
    cssGA = GASpliceSites.load_data_tsv(cssFile)
    authssGA = GASpliceSites.load_data_tsv(authssFile)
    cssGASpliceSites = GASpliceSites(cssGA)
    authGASpliceSites = GASpliceSites(authssGA)

    M = 10
    N = 9 # 9-mers
    initPopulation = randomSpliceSitesPopulation(M, N)
    print(initPopulation)

    # authThread = GASpliceSitesThread(authGASpliceSites, initPopulation, genCount = 10)
    cssThread = GASpliceSitesThread(cssGASpliceSites, initPopulation, genCount = 10)

    # authThread.start()
    cssThread.start()

    # authThread.join()
    cssThread.join()
    cssGABase = cssThread.gaBase

    genFitness = [gen._bestFitness for gen in cssGABase._generations]
    bestFitness_all = max(genFitness)
    bestGenArr = filter(lambda g: g._bestFitness == bestFitness_all, cssGABase._generations)
    
    print("")
    
    for bestGen in bestGenArr:
        print("BEST: %s" % (bestGen))

    bestGen = bestGenArr[0]
    lastGen = cssGABase._generations[-1]
    bestGenMatch_withCss = check_match(bestGen._population, cssGA)
    bestGenMatch_withAuth = check_match(bestGen._population, authssGA)
    print("bestGenMatch_withCss: %s" % str(bestGenMatch_withCss))
    print("bestGenMatch_withAuth: %s" % str(bestGenMatch_withAuth))




    # cssGAThread = main_inst(cssPwm, M, N, GEN_COUNT=100)
    # authssGAThread = main_inst(authssPwm, M, N, GEN_COUNT=100)
    # cssData = read(cssFile)
    # authssData = read(authssFile)
    # scoreFunction(pop, cssPwm, authssPwm)
    #     cssScore
    #     authScore
    # cssGAThread.start
    # authssGAThread.start
    # join
    # join

# def main_inst(knapsack, M, N, GEN_COUNT=10):
#     initPopulation = randomPopulation(M, N, 2)
#     print(knapsack)

#     gen0 = Generation(initPopulation)
#     recombine = lambda population: Selections.rouletteWheel(initPopulation, knapsack.fitness)
#     evolution = EvolutionBasic(select = recombine, crossover = Crossovers.two_point, mutate = Mutations.bool_flip)
#     gaBase = GABase(evolution, gen0, knapsack.fitness)
#     gaBase.execute(maxGens=GEN_COUNT)
#     genFitness = [gen._bestFitness for gen in gaBase._generations]
#     bestFitness_all = max(genFitness)
#     bestGenArr = filter(lambda g: g._bestFitness == bestFitness_all, gaBase._generations)
#     print ""
#     for bestGen in bestGenArr:
#         print("BEST: %s, profit: %s, weight: %s" % (bestGen, knapsack.profit(bestGen._bestSolution), knapsack.weight(bestGen._bestSolution)))
#     print "SOLUTION: %s, solutionFitness: %s, solutionProfit: %s, solutionWeight: %s" %(knapsack._solution, 
#         knapsack.fitness(knapsack._solution), knapsack.profit(knapsack._solution), knapsack.weight(knapsack._solution))

if __name__ == '__main__':
    main()