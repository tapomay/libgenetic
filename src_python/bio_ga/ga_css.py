'''
create pwmCss, pwmAuth
define fitness
initPopulation as randomized
scoring

'''

import random
import threading
import time
from libgenetic.libgenetic import EvolutionBasic, Selections, Crossovers, Mutations, Generation, GABase
import numpy as np

BASES_MAP = {0:'A', 1:'C', 2:'G', 3:'T'}
BASE_ODDS_MAP = {'A':0.28, 'C': 0.22, 'G': 0.22, 'T': 0.28}

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
                    raise Exception("Unrecognized symbol in input: %s; symSet: %s; row: %s" % (sym, symbolSet, str(row)))

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

    @staticmethod
    def testSelf_3prime():
        dataArr=[
            ['C', 'T', 'G', 'G', 'T', 'A', 'A', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['C', 'A', 'A', 'G', 'T', 'A', 'A', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['A', 'A', 'G', 'G', 'T', 'A', 'T', 'A', 'T', 'A', 'A', 'G', 'T'],
            ['A', 'G', 'T', 'G', 'T', 'A', 'A', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['T', 'T', 'G', 'G', 'T', 'A', 'A', 'A', 'A', 'A', 'A', 'G', 'T'],
            ['T', 'G', 'G', 'G', 'T', 'A', 'A', 'G', 'G', 'A', 'A', 'G', 'T'],
            ['C', 'A', 'G', 'G', 'T', 'G', 'A', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['A', 'G', 'G', 'G', 'T', 'A', 'A', 'T', 'G', 'A', 'A', 'G', 'T'],
            ['T', 'A', 'G', 'G', 'T', 'A', 'T', 'T', 'G', 'A', 'A', 'G', 'T'],
            ['C', 'A', 'G', 'G', 'T', 'A', 'A', 'A', 'A', 'A', 'A', 'G', 'T'],
            ['A', 'A', 'G', 'G', 'T', 'G', 'T', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['A', 'A', 'G', 'G', 'T', 'A', 'A', 'G', 'A', 'A', 'A', 'G', 'T'],
            ['T', 'A', 'G', 'G', 'T', 'A', 'A', 'T', 'A', 'A', 'A', 'G', 'T'],
            ['T', 'T', 'T', 'G', 'T', 'G', 'A', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['C', 'A', 'G', 'G', 'T', 'A', 'T', 'A', 'C', 'A', 'A', 'G', 'T'],
            ['T', 'C', 'T', 'G', 'T', 'A', 'A', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['G', 'A', 'G', 'G', 'T', 'A', 'A', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['A', 'A', 'G', 'G', 'T', 'A', 'A', 'A', 'G', 'A', 'A', 'G', 'T'],
            ['C', 'A', 'G', 'G', 'T', 'A', 'A', 'G', 'A', 'A', 'A', 'G', 'T'],
            ['A', 'C', 'A', 'G', 'T', 'A', 'A', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['A', 'T', 'G', 'G', 'T', 'A', 'A', 'G', 'G', 'A', 'A', 'G', 'T']
        ]
        symbolSet = set(['A', 'C', 'G', 'T'])
        symbolOddsMap = {'A':0.28, 'C': 0.22, 'G': 0.22, 'T': 0.28}
        pwm = PWM(dataArr, symbolSet, symbolOddsMap)
        print(pwm.pwmMatrix)

        testSolution = "CTGGTAAGTAAGT"
        testScore = pwm.score(testSolution)
        print(testScore)
        return pwm

class EI5pSpliceSitesGAModel:
    '''
        Wrapper for 5 prime splice site 9mers with PWM score fitness
    '''
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
                if line:
                    bases = line.split('\t')
                    ret.append(bases)
        return ret

    def __init__(self, nmerArr):
        self._nmerArr = nmerArr
        self._pwm = self._computePwm(nmerArr)

    def _computePwm(self, nmerArr):
        symbolSet = set(['A', 'C', 'G', 'T'])
        symbolOddsMap = BASE_ODDS_MAP
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
        flipValue = BASES_MAP[randomBase] # BASES_MAP = {0:'A', 1:'C', 2:'G', 3:'T'}
        while flipValue == baseValue:
            randomBase = random.randint(0, 3)
            flipValue = BASES_MAP[randomBase]
        return flipValue

    def crossover(self, sol1, sol2):
        sol1Len = len(sol1)
        site = random.randint(1, sol1Len - 2)
        # prevent crossover through GT at sites 3, 4, first and last bases
        negatives = [0, 3, 4, 8]
        while site in negatives:
            site = random.randint(0, sol1Len) 
        return Crossovers.two_point(sol1, sol2, site = site)

    def mutate(self, solution):
        return Mutations.provided_flip(solution = solution, flipProvider = self.baseFlip, negativeSites = [3,4])

    def __str__(self):
        pass
        # return "P: %s\nW: %s\nC: %s\nS: %s" % (str(self._profits), str(self._weights), str(self._capacity), str(self._solution))

class GASpliceSitesThread(threading.Thread):
    def __init__(self, gASpliceSites, initPopulation, genCount,
        crossoverProbability = 0.1, mutationProbability = 0.1):
        threading.Thread.__init__(self)
        self._gASpliceSites = gASpliceSites
        self._initPopulation = initPopulation
        self._genCount = genCount
        self._crossoverProbability = crossoverProbability
        self._mutationProbability = mutationProbability
        self._gaBase = None

    def run(self):
        gen0 = Generation(self._initPopulation)
        recombine = lambda population: Selections.rouletteWheel(population, self._gASpliceSites.fitness)
        crossover = self._gASpliceSites.crossover
        mutator = self._gASpliceSites.mutate
        evolution = EvolutionBasic(select = recombine, crossover = crossover, mutate = mutator,
            crossoverProbability = self._crossoverProbability, 
            mutationProbability = self._mutationProbability)
        gaBase = GABase(evolution, gen0, self._gASpliceSites.fitness)
        gaBase.execute(maxGens=self._genCount)
        self._gaBase = gaBase

    @property
    def gaBase(self):
        return self._gaBase

def random5primeSpliceSitesPopulation(M, N, cardinality=4):
    randomgen = lambda: random.randint(0, cardinality - 1)
    basesMap = BASES_MAP # BASES_MAP = {0:'A', 1:'C', 2:'G', 3:'T'}
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

class MatchUtils:

    @staticmethod
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

    @staticmethod
    def find_best_gens(gaBase):
        genFitness = [gen._bestFitness for gen in gaBase._generations]
        bestFitness_all = max(genFitness)
        bestGenArr = filter(lambda g: g._bestFitness == bestFitness_all, gaBase._generations)
        return bestGenArr

    @staticmethod
    def match_stat(gaBase, authssData, cssData):
        lastGen = gaBase._generations[-1]
        bestGens = MatchUtils.find_best_gens(gaBase)
        # bestgen_scoreCss = [MatchUtils.check_match(gen._population, cssData) for gen in bestGens]
        # bestgen_scoreCss = max(bestgen_scoreCss)
        # bestgen_scoreAuth = [MatchUtils.check_match(gen._population, authssData) for gen in bestGens]
        # bestgen_scoreAuth = max(bestgen_scoreAuth)
        bestgen_scoreCss = -float('inf')
        bestgen_scoreAuth = -float('inf')
        bestgen_genCss = -1
        bestgen_genAuth = -1

        for gen in bestGens:
            scoreCss = MatchUtils.check_match(gen._population, cssData)
            if scoreCss > bestgen_scoreCss:
                bestgen_scoreCss = scoreCss
                bestgen_genCss = gen._genIndex
            scoreAuth = MatchUtils.check_match(gen._population, authssData)
            if scoreAuth > bestgen_scoreAuth:
                bestgen_scoreAuth = scoreAuth
                bestgen_genAuth = gen._genIndex

        lastgen_scoreCss = MatchUtils.check_match(lastGen._population, cssData)
        lastgen_scoreAuth = MatchUtils.check_match(lastGen._population, authssData)
        return (bestgen_scoreCss, bestgen_scoreAuth, bestgen_genCss, bestgen_genAuth, lastgen_scoreCss, lastgen_scoreAuth)

def main(cssFile = 'data/dbass-prats/CrypticSpliceSite.tsv', 
    authssFile = 'data/hs3d/Exon-Intron_5prime/EI_true_9.tsv', 
    generationSize = 10, genCount = 10,
    crossoverProbability = 0.1, mutationProbability = 0.1):
    '''
        Compare AuthPWMGA <-> CSSPWMGA
        AuthPWMGA genN should predominantly carry stochastic properties of authentic SS data
        CSSPWMGA genN should predominantly carry stochastic properties of cryptic SS data
    '''
    cssGAData = EI5pSpliceSitesGAModel.load_data_tsv(cssFile)
    authssGAData = EI5pSpliceSitesGAModel.load_data_tsv(authssFile)
    cssGASpliceSites = EI5pSpliceSitesGAModel(cssGAData)
    authGASpliceSites = EI5pSpliceSitesGAModel(authssGAData)

    M = generationSize
    N = 9 # 9-mers
    initPopulation = random5primeSpliceSitesPopulation(M, N)
    print(initPopulation)

    authThread = GASpliceSitesThread(authGASpliceSites, initPopulation, genCount = genCount,
        crossoverProbability = 0.1, mutationProbability = 0.1)
    cssThread = GASpliceSitesThread(cssGASpliceSites, initPopulation, genCount = genCount,
        crossoverProbability = 0.1, mutationProbability = 0.1)

    cssThread.start()
    cssThread.join()
    cssGABase = cssThread.gaBase

    authThread.start()
    authThread.join()
    authGABase = authThread.gaBase

    stats = []
    stats.append(['TRAINER', 'bestgen_scoreCss', 'bestgen_scoreAuth', 'bestgen_genCss', 'bestgen_genAuth', 'lastgen_scoreCss', 'lastgen_scoreAuth'])
    
    (bestgen_scoreCss, bestgen_scoreAuth, bestgen_genCss, bestgen_genAuth, lastgen_scoreCss, lastgen_scoreAuth) = \
        MatchUtils.match_stat(cssGABase, authssGAData, cssGAData)
    print("\nCSS GAStats:")
    print("BESTGEN: cssGen_X_cssData: %s" % str(bestgen_scoreCss))
    print("BESTGEN: cssGen_X_authData: %s" % str(bestgen_scoreAuth))
    print("BESTGENIDX: cssGen_X_cssData: %s" % str(bestgen_genCss))
    print("BESTGENIDX: cssGen_X_authData: %s" % str(bestgen_genAuth))
    print("LASTGEN: cssGen_X_cssData: %s" % str(lastgen_scoreCss))
    print("LASTGEN: cssGen_X_authData: %s" % str(lastgen_scoreAuth))
    stats.append(['cssGABase', bestgen_scoreCss, bestgen_scoreAuth, bestgen_genCss, bestgen_genAuth, lastgen_scoreCss, lastgen_scoreAuth])

    (bestgen_scoreCss, bestgen_scoreAuth, bestgen_genCss, bestgen_genAuth, lastgen_scoreCss, lastgen_scoreAuth) = \
        MatchUtils.match_stat(authGABase, authssGAData, cssGAData)
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

    import os
    import argparse
    parser = argparse.ArgumentParser(description='libgenetic implementation for Splice Site evolution using PWM')
    parser.add_argument('--gen_count', type=int, help='generation count', required=True)
    parser.add_argument('--gen_size', type=int, help='generation size', required=True)
    parser.add_argument('--xover_prob', type=float, help='crossover probability', default=0.7)
    parser.add_argument('--mut_prob', type=float, help='mutation probability', default=0.1)
    parser.add_argument('--css_file', help='path to css tsv data file', default='%s/data/dbass-prats/CrypticSpliceSite.tsv' % os.getcwd())
    parser.add_argument('--authss_file', help='path to authss tsv data file', default='%s/data/hs3d/Exon-Intron_5prime/EI_true_9.tsv' % os.getcwd())
    args = parser.parse_args()

    main(cssFile = args.css_file, authssFile = args.authss_file, 
    generationSize = args.gen_size, genCount = args.gen_count,
    crossoverProbability = args.xover_prob, mutationProbability = args.mut_prob)