import random
import threading
import time
from libgenetic.libgenetic import EvolutionBasic, Selections, Crossovers, Mutations, Generation, GABase
import numpy as np
from ga_css import random5primeSpliceSitesPopulation, EI5pSpliceSitesGAModel, GASpliceSitesThread, MatchUtils, PWM, BASES_MAP, BASE_ODDS_MAP
from ga_css_altern import mergeSSData

cssFile = '../../data/dbass-prats/CrypticSpliceSite.tsv'
authssFile = '../../data/hs3d/Exon-Intron_5prime/EI_true_9.tsv'
alternssFile = '../../data/dbass-prats/NeighboringSpliceSite.tsv'
generationSize = 10
genCount = 10
crossoverProbability = 0.1
mutationProbability = 0.1

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
    MatchUtils.match_stat(combinedGABase, combinedGAData, alternssGAData)
print("\nAltern GAStats:")
print("BESTGEN: alternGen_X_alternData: %s" % str(bestgen_scoreAltern))
print("BESTGEN: alternGen_X_combinedData: %s" % str(bestgen_scoreCombined))
print("BESTGENIDX: alternGen_X_alternData: %s" % str(bestgen_genAltern))
print("BESTGENIDX: alternGen_X_combinedData: %s" % str(bestgen_genCombined))
print("LASTGEN: alternGen_X_alternData: %s" % str(lastgen_scoreAltern))
print("LASTGEN: alternGen_X_combinedData: %s" % str(lastgen_scoreCombined))
stats.append(['alternGABase', bestgen_scoreAltern, bestgen_scoreCombined, bestgen_genAltern, bestgen_genCombined, lastgen_scoreAltern, lastgen_scoreCombined])

(bestgen_scoreAltern, bestgen_scoreCombined, bestgen_genAltern, bestgen_genCombined, lastgen_scoreAltern, lastgen_scoreCombined) = \
    MatchUtils.match_stat(authGABase, combinedGAData, alternssGAData)
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
