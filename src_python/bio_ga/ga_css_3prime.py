
import random
import threading
import time
from libgenetic.libgenetic import EvolutionBasic, Selections, Crossovers, Mutations, Generation, GABase
import numpy as np
from ga_css_5prime import EI5pSpliceSitesGAModel, GASpliceSitesThread, MatchUtils, PWM, BASES_MAP, BASE_ODDS_MAP

class IE3pSpliceSitesGAModel(EI5pSpliceSitesGAModel):
    '''
        Wrapper for 3 prime splice site 13mers with PWM score fitness
    '''
    def __init__(self, nmerArr):
        self._nmerArr = nmerArr
        self._pwm = self._computePwm(self._nmerArr)

    def _computePwm(self, nmerArr):
        symbolSet = set(['A', 'C', 'G', 'T'])
        symbolOddsMap = BASE_ODDS_MAP
        pwm = PWM(nmerArr, symbolSet, symbolOddsMap)
        return pwm

    def isValid13merSpliceSite(self, nmer):
        if len(nmer) != 13:
            raise Exception("dimension mismatch")
        return nmer[10] == 'A' and nmer[11] == 'G'

    def fitness(self, nmer):
        baseScore = self._pwm.score(nmer)
        penalty = 0
        if not self.isValid13merSpliceSite(nmer):
            # print("Invalid nmer: %s" % nmer)
            penalty = 13 * 1000 #TODO: max logodds score; log2(1 / min(symOdds))
        ret = baseScore - penalty
        return ret

    def crossover(self, sol1, sol2):
        sol1Len = len(sol1)
        site = random.randint(1, sol1Len - 2)
        # prevent crossover through AG at sites 10, 11, first and last bases
        negatives = [10, 11, 0, 12]
        while site in negatives:
            site = random.randint(0, sol1Len)
        return Crossovers.two_point(sol1, sol2, site = site)

    @staticmethod
    def crossover_1p(sol1, sol2):
        sol1Len = len(sol1)
        site = random.randint(1, sol1Len - 2)
        negatives = [10, 11] # prevent crossover through AG at sites 10, 11
        site = Crossovers.pick_random_site(rangeLen = 13, negativeSites = negatives)
        ret = Crossovers.one_point(sol1, sol2, site = site)
        return ret

    @staticmethod
    def crossover_2p(sol1, sol2):
        sol1Len = len(sol1)
        site = random.randint(1, sol1Len - 2)
        negatives = [10, 11] # prevent crossover through AG at sites 10, 11
        site1 = Crossovers.pick_random_site(rangeLen = 13, negativeSites = negatives) 
        negatives = negatives + [site1]
        site2 = Crossovers.pick_random_site(rangeLen = 13, negativeSites = negatives) 
        ret = Crossovers.two_point(sol1, sol2, site1 = site1, site2 = site2)
        return ret

    @staticmethod
    def crossover_uniform(sol1, sol2, swap_prob = 0.5):
        negatives = [10, 11] # prevent crossover through AG at sites 10, 11
        ret = Crossovers.uniform(sol1, sol2, swap_prob = swap_prob, negativeSites = negatives)
        return ret

    @staticmethod
    def crossover_uniform_orderbased(sol1, sol2):
        negatives = [10, 11] # prevent crossover through AG at sites 10, 11
        ret = Crossovers.uniform_orderbased(sol1, sol2, negativeSites = negatives)
        return ret

    def mutate(self, solution):
        return Mutations.provided_flip(solution = solution, flipProvider = self.baseFlip, negativeSites = [10,11])

    def __str__(self):
        pass
        # return "P: %s\nW: %s\nC: %s\nS: %s" % (str(self._profits), str(self._weights), str(self._capacity), str(self._solution))

def random3primeSpliceSitesPopulation(M, N, cardinality=4):
    randomgen = lambda: random.randint(0, cardinality - 1)
    basesMap = BASES_MAP # BASES_MAP = {0:'A', 1:'C', 2:'G', 3:'T'}
    ret = []
    for i in range(M):
        sol = []
        for j in range(N):
            if j == 10:
                bit = 0 #'A'
            elif j == 11:
                bit = 2 #'G'
            else:
                bit = randomgen()
            sol.append(basesMap[bit])
        ret.append(sol)
    return ret

def main(cssFile = 'data/dbass/css_3prime.tsv', 
    authssFile = 'data/hs3d/Intron-Exon_3prime/authss_3prime.tsv', 
    generationSize = 10, genCount = 10,
    crossoverProbability = 0.1, mutationProbability = 0.1):
    cssGAData = IE3pSpliceSitesGAModel.load_data_tsv(cssFile)
    authssGAData = IE3pSpliceSitesGAModel.load_data_tsv(authssFile)
    cssGASpliceSites = IE3pSpliceSitesGAModel(cssGAData)
    authGASpliceSites = IE3pSpliceSitesGAModel(authssGAData)

    M = generationSize
    N = 13 # 13-mers
    initPopulation = random3primeSpliceSitesPopulation(M, N)
    print(initPopulation)

    recombine_provider = lambda gaModel: lambda population: Selections.ranked(population, gaModel.fitness)
    crossover_provider = lambda gaModel: EI5pSpliceSitesGAModel.crossover_uniform_orderbased

    authThread = GASpliceSitesThread(authGASpliceSites, initPopulation, genCount = genCount,
        crossoverProbability = crossoverProbability, mutationProbability = mutationProbability, 
        recombine_provider = recombine_provider,
        crossover_provider = crossover_provider)
    cssThread = GASpliceSitesThread(cssGASpliceSites, initPopulation, genCount = genCount,
        crossoverProbability = crossoverProbability, mutationProbability = mutationProbability, 
        recombine_provider = recombine_provider,
        crossover_provider = crossover_provider)

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
    parser.add_argument('--css_file', help='path to css tsv data file', default='%s/data/dbass/css_3prime.tsv' % os.getcwd())
    parser.add_argument('--authss_file', help='path to authss tsv data file', default='%s/data/hs3d/Intron-Exon_3prime/authss_3prime.tsv' % os.getcwd())
    args = parser.parse_args()

    main(cssFile = args.css_file, authssFile = args.authss_file, 
    generationSize = args.gen_size, genCount = args.gen_count,
    crossoverProbability = args.xover_prob, mutationProbability = args.mut_prob)