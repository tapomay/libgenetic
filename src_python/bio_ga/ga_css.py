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
from libgenetic.pwm import PWM
import numpy as np

BASES_MAP = {0:'A', 1:'C', 2:'G', 3:'T'}
BASE_ODDS_MAP = {'A':0.28, 'C': 0.22, 'G': 0.22, 'T': 0.28}


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

    def crossover_1p(self, sol1, sol2):
        sol1Len = len(sol1)
        site = random.randint(1, sol1Len - 2)
        # prevent crossover through GT at sites 3, 4, first and last bases
        negatives = [3, 4]
        site = Crossovers.pick_random_site(rangeLen = 9, negativeSites = negatives)
        ret = Crossovers.one_point(sol1, sol2, site = site)
        return ret

    def crossover_2p(self, sol1, sol2):
        sol1Len = len(sol1)
        site = random.randint(1, sol1Len - 2)
        # prevent crossover through GT at sites 3, 4, first and last bases
        negatives = [3, 4]
        site1 = Crossovers.pick_random_site(rangeLen = 9, negativeSites = negatives) 
        negatives = negatives + [site1]
        site2 = Crossovers.pick_random_site(rangeLen = 9, negativeSites = negatives) 
        ret = Crossovers.two_point(sol1, sol2, site1 = site1, site2 = site2)
        return ret

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
        recombine = lambda population: Selections.ranked(population, self._gASpliceSites.fitness)
        crossover = self._gASpliceSites.crossover_1p
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