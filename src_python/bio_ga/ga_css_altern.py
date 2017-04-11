'''
create pwmCss, pwmAuth, pwmAltern
define fitness
initPopulation as randomized
scoring

'''

import random
import threading
import time
from libgenetic.libgenetic import EvolutionBasic, Selections, Crossovers, Mutations, Generation, GABase
import numpy as np
from ga_css import random5primeSpliceSitesPopulation, EI5pSpliceSitesGAModel, GASpliceSitesThread, MatchUtils, PWM, BASES_MAP, BASE_ODDS_MAP

def mergeSSData(data1, data2):
    npdata1 = np.array(data1)
    npdata2 = np.array(data2)
    retnp = np.concatenate((npdata1, npdata2))
    ret = retnp.tolist()
    return ret

def main(cssFile = 'data/dbass-prats/CrypticSpliceSite.tsv', 
    authssFile = 'data/hs3d/Exon-Intron_5prime/EI_true_9.tsv', 
    alternssFile = None, 
    generationSize = 10, genCount = 10,
    crossoverProbability = 0.1, mutationProbability = 0.1):

    '''
        Compare (AuthGA + CSSGA) <-> AlternGA
        Create (Auth+CSS)PWMGA. Its genN should predominantly carry stochastic properties of auth + css data
        Create AlternPWMGA. Its genN should predominantly carry stochastic properties of altern data
    '''
    cssGAData = EI5pSpliceSitesGAModel.load_data_tsv(cssFile)
    authssGAData = EI5pSpliceSitesGAModel.load_data_tsv(authssFile)
    alternssGAData = EI5pSpliceSitesGAModel.load_data_tsv(alternssFile)

    combinedGAData = mergeSSData(authssGAData, cssGAData)
    
    combinedGASpliceSites = EI5pSpliceSitesGAModel(combinedGAData)
    alternGASpliceSites = EI5pSpliceSitesGAModel(alternssGAData)

    M = generationSize
    N = 9 # 9-mers
    initPopulation = random5primeSpliceSitesPopulation(M, N)
    print(initPopulation)

    combinedDataThread = GASpliceSitesThread(combinedGASpliceSites, initPopulation, genCount = genCount,
        crossoverProbability = 0.1, mutationProbability = 0.1)
    alternThread = GASpliceSitesThread(alternGASpliceSites, initPopulation, genCount = genCount,
        crossoverProbability = 0.1, mutationProbability = 0.1)

    combinedDataThread.start()
    combinedDataThread.join()
    combinedGABase = combinedDataThread.gaBase

    alternThread.start()
    alternThread.join()
    alternGABase = alternThread.gaBase

    stats = []
    stats.append(['TRAINER', 'bestgen_scoreAltern', 'bestgen_scoreCombined', 'bestgen_genAltern', 'bestgen_genCombined', 'lastgen_scoreAltern', 'lastgen_scoreCombined'])
    
    (bestgen_scoreAltern, bestgen_scoreCombined, bestgen_genAltern, bestgen_genCombined, lastgen_scoreAltern, lastgen_scoreCombined) = \
        MatchUtils.match_stat(alternGABase, combinedGAData, alternssGAData)
    print("\nAltern GAStats:")
    print("BESTGEN: alternGen_X_alternData: %s" % str(bestgen_scoreAltern))
    print("BESTGEN: alternGen_X_combinedData: %s" % str(bestgen_scoreCombined))
    print("BESTGENIDX: alternGen_X_alternData: %s" % str(bestgen_genAltern))
    print("BESTGENIDX: alternGen_X_combinedData: %s" % str(bestgen_genCombined))
    print("LASTGEN: alternGen_X_alternData: %s" % str(lastgen_scoreAltern))
    print("LASTGEN: alternGen_X_combinedData: %s" % str(lastgen_scoreCombined))
    stats.append(['alternGABase', bestgen_scoreAltern, bestgen_scoreCombined, bestgen_genAltern, bestgen_genCombined, lastgen_scoreAltern, lastgen_scoreCombined])

    (bestgen_scoreAltern, bestgen_scoreCombined, bestgen_genAltern, bestgen_genCombined, lastgen_scoreAltern, lastgen_scoreCombined) = \
        MatchUtils.match_stat(combinedGABase, combinedGAData, alternssGAData)
    print("\nCOMBINED GAStats:")
    print("BESTGEN: alternGen_X_alternData: %s" % str(bestgen_scoreAltern))
    print("BESTGEN: alternGen_X_combinedData: %s" % str(bestgen_scoreCombined))
    print("BESTGENIDX: alternGen_X_alternData: %s" % str(bestgen_genAltern))
    print("BESTGENIDX: alternGen_X_combinedData: %s" % str(bestgen_genCombined))
    print("LASTGEN: alternGen_X_alternData: %s" % str(lastgen_scoreAltern))
    print("LASTGEN: alternGen_X_combinedData: %s" % str(lastgen_scoreCombined))
    stats.append(['combinedGABase', bestgen_scoreAltern, bestgen_scoreCombined, bestgen_genAltern, bestgen_genCombined, lastgen_scoreAltern, lastgen_scoreCombined])

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
    parser.add_argument('--altss_file', help='path to alternative ss tsv data file', default='%s/data/dbass-prats/NeighboringSpliceSite.tsv' % os.getcwd())
    args = parser.parse_args()

    main(cssFile = args.css_file, authssFile = args.authss_file, alternssFile = args.altss_file,
    generationSize = args.gen_size, genCount = args.gen_count,
    crossoverProbability = args.xover_prob, mutationProbability = args.mut_prob)