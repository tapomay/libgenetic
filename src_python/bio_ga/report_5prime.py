from tabulate import tabulate
from collections import OrderedDict
import ga_css_5prime as ga

from ga_css_5prime import EI5pSpliceSitesGAModel, GASpliceSitesThread, MatchUtils, PWM, BASES_MAP, BASE_ODDS_MAP, random5primeSpliceSitesPopulation
from libgenetic.libgenetic import Selections, Crossovers
import numpy as np

'''
Pseudo code:
datasets = [Auth/A, Crypt/C, Neighbor/N, Auth + Crypt/AC]
run_counts = [10,20,50,100]

Parametrize: gencount, gensizem dataset
Execute each parameter combination for each run_counts times.

runCount = 100
for each gensize:
	for each gencount:
		for each dataset in datasets:
			spec = (gensize, gencount, dataset)
			specFitnessScores = []
			specMatchScores = []
			for run_count times:
				specGABase = execute(spec)
				specGABaseFitness = bestFitness(specGABase)
				specFitnessScores.append(specGABaseFitness)
				specGABaseMatch = match_stat_bestgen(specGABase, A/C/N/AC) (A X C, C X A, N X AC)
				specMatchScores.append(specGABaseMatch)
				specGABaseBest = compare_best(specGABase, specGABaseBest) if genCount >=1000
			specRunCountFitnessStats[spec] = histogram(specFitnessScores)
			specRunCountMatchStats[spec] = histogram(specMatchScores)
			graph : specGABaseBest.generations -> specGABaseBest.fitness

'''

CSS_5PRIME_FILE = 'data/dbass-prats/CrypticSpliceSite.tsv'
AUTH_5PRIME_FILE = 'data/hs3d/Exon-Intron_5prime/EI_true_9.tsv'
NEIGH_5PRIME_FILE = 'data/dbass-prats/NeighboringSpliceSite.tsv'

class DatasetSpec:
	def __init__(self, name, gaModel, compareModel):
		self.name = name
		self.gaModel = gaModel
		self.compareModel = compareModel

class Composite5pGA:

	def __init__(self):
		self.css5pData = EI5pSpliceSitesGAModel.load_data_tsv(CSS_5PRIME_FILE)
		self.authss5pData = EI5pSpliceSitesGAModel.load_data_tsv(AUTH_5PRIME_FILE)	
		self.nss5pData = EI5pSpliceSitesGAModel.load_data_tsv(NEIGH_5PRIME_FILE)
		self.ac5pData = EI5pSpliceSitesGAModel.mergeSSData(self.authss5pData, self.css5pData)

		self.css5pGAModel = EI5pSpliceSitesGAModel(self.css5pData)
		self.auth5pGAModel = EI5pSpliceSitesGAModel(self.authss5pData)
		self.nss5pGAModel = EI5pSpliceSitesGAModel(self.nss5pData)
		self.ac5pGAModel = EI5pSpliceSitesGAModel(self.ac5pData)

	def getspecs(self):
		ret = {}
		ret["auth"] = DatasetSpec("auth", self.auth5pGAModel, self.css5pGAModel)
		ret["css"] = DatasetSpec("css", self.css5pGAModel, self.auth5pGAModel)
		ret["neighbor"] = DatasetSpec("neighbor", self.nss5pGAModel, self.ac5pGAModel)
		return ret

class MultirunExecStat:
	
	def __init__(self, keyTuple, runIdxPerfArr, bestGABase):
		self.scoreSelfHisto = MultirunExecStat.histogram([perf[0] for perf in runIdxPerfArr])
		self.scoreCompeteHisto = MultirunExecStat.histogram([perf[1] for perf in runIdxPerfArr])
		self.scoreFitnessHisto = MultirunExecStat.histogram([perf[2] for perf in runIdxPerfArr])

		self.scoreSelfMean = np.mean([perf[0] for perf in runIdxPerfArr])
		self.scoreCompeteMean = np.mean([perf[1] for perf in runIdxPerfArr])
		self.scoreFitnessMean = np.mean([perf[2] for perf in runIdxPerfArr])
		self.bestGABase = bestGABase

	@staticmethod	
	def histogram(data):
		from collections import Counter
		ret = Counter(data)
		return ret.items()

def waitForCompletion(gaModel, initPopulation, genCount, select_oper, xover_oper, cprob, mprob):
    gaThread = GASpliceSitesThread(gaModel, initPopulation, genCount = genCount,
        crossoverProbability = cprob, mutationProbability = mprob,
        crossover = xover_oper, recombine = None)
    select = select_oper(gaThread)
    gaThread._recombine = select

    gaThread.start()
    gaThread.join()
    gaBase = gaThread.gaBase
    return gaBase


def execute(selection_selector, xover, dataspecs):
	'''
	for each gensize:
		for each gencount:
			for each dataset in datasets:
				spec = (gensize, gencount, dataset)
				specFitnessScores = []
				specMatchScores = []
				for run_count times:
					specGABase = execute(spec)
					specGABaseFitness = bestFitness(specGABase)
					specFitnessScores.append(specGABaseFitness)
					specGABaseMatch = match_stat_bestgen(specGABase, A/C/N/AC) (A X C, C X A, N X AC)
					specMatchScores.append(specGABaseMatch)
					specGABaseBest = compare_best(specGABase, specGABaseBest) if genCount >=1000
				specRunCountFitnessStats[spec] = histogram(specFitnessScores)
				specRunCountMatchStats[spec] = histogram(specMatchScores)
				graph : specGABaseBest.generations -> specGABaseBest.fitness
	'''
	generationSizes = [10, 20]
	generationCounts = [10, 20, 50, 100, 1000]
	runCount = 100
	execStat = {}
	for gensize in generationSizes:
		M = gensize
		N = 9 # 9-mers
		initPopulation = random5primeSpliceSitesPopulation(M = M, N = N)
		for gencount in generationCounts:
			for dataspec in dataspecs:
				try:
					runIdxPerfArr = []
					bestGABase = None
					bestGAFitness = -float('inf')
					for runIdx in range(runCount):
						print("EXECUTION(gensize: %d, gencount: %d, dataspec.name: %s, runIdx: %d)" % (gensize, gencount, dataspec.name, runIdx))
						gaBase = waitForCompletion(dataspec.gaModel, initPopulation, genCount=gencount, select_oper = selection_selector, xover_oper = xover, cprob = 0.7, mprob = 0.1)
						
						((bestgen_scoreSelf, bestgen_scoreCompete), (bestgen_genSelf, bestgen_genCompete), fitness) = \
							MatchUtils.match_stat_2d(gaBase=gaBase, selfData=dataspec.gaModel.rawdata, competeData=dataspec.compareModel.rawdata)
						runIdxPerfArr.append((bestgen_scoreSelf, bestgen_scoreCompete, fitness))
						# print("RUN: %s: %s" % (str((gensize, gencount, dataspec.name, runIdx)), str(runIdxPerfArr)))
						if fitness > bestGAFitness: #check for runIdx that gives gaBase with healthiest generation
							bestGAFitness = fitness
							bestGABase = gaBase
					execStat[(gensize, gencount, dataspec.name)] = MultirunExecStat((gensize, gencount, dataspec.name), runIdxPerfArr, bestGABase)
				except BaseException as e:
					print("EXEC %s FAILED: %s" % (str((gensize, gencount, dataspec.name)), e))
					execStat[(gensize, gencount, dataspec.name)] = e
	return execStat

def main():

	composite = Composite5pGA()
	dataspecs = composite.getspecs().values()

	selections_selector = {"rouletteWheel": lambda gaThread: lambda population: Selections.rouletteWheel(population, gaThread._gaModel.fitness), 
			"ranked": lambda gaThread: lambda population: Selections.ranked(population, gaThread._gaModel.fitness), 
			"tournament": lambda gaThread: lambda population: Selections.tournament(population, gaThread._gaModel.fitness)}

	crossovers = {"crossover_1p": EI5pSpliceSitesGAModel.crossover_1p, "crossover_2p": EI5pSpliceSitesGAModel.crossover_2p, "crossover_uniform": EI5pSpliceSitesGAModel.crossover_uniform}

	sel = selections_selector["rouletteWheel"]
	s_name = "rouletteWheel"
	perfMap = dict()

	for x_name, xover in crossovers.items():
		execStatMap = execute(sel, xover, dataspecs)
		print("INITIATING EXECUTION FOR: %s" % str((s_name, x_name)))
		perfMap[(s_name, x_name)] = execStatMap
	
	xover = crossovers["crossover_1p"]
	x_name = "crossover_1p"
	for s_name, sel in selections_selector.items():
		print("INITIATING EXECUTION FOR: %s" % str((s_name, x_name)))
		execStatMap = execute(sel, xover, dataspecs)
		perfMap[(s_name, x_name)] = execStatMap

	return perfMap

if __name__ == '__main__':
	main()