from tabulate import tabulate
from collections import OrderedDict
import ga_css_3prime as ga

# from ga_css_5prime import EI5pSpliceSitesGAModel, GASpliceSitesThread, MatchUtils, PWM, BASES_MAP, BASE_ODDS_MAP, random5primeSpliceSitesPopulation
from ga_css_3prime import IE3pSpliceSitesGAModel, random3primeSpliceSitesPopulation, GASpliceSitesThread, MatchUtils

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

CSS_3PRIME_FILE = 'data/dbass/css_3prime.tsv'
AUTH_3PRIME_FILE = 'data/hs3d/Intron-Exon_3prime/authss_3prime.tsv'
# NEIGH_5PRIME_FILE = 'data/dbass-prats/NeighboringSpliceSite.tsv'

class DatasetSpec:
	def __init__(self, name, gaModel, compareModel):
		self.name = name
		self.gaModel = gaModel
		self.compareModel = compareModel

class Composite3pGA:

	def __init__(self):
		self.css3pData = IE3pSpliceSitesGAModel.load_data_tsv(CSS_3PRIME_FILE)
		self.authss3pData = IE3pSpliceSitesGAModel.load_data_tsv(AUTH_3PRIME_FILE)	

		self.css3pGAModel = IE3pSpliceSitesGAModel(self.css3pData)
		self.auth3pGAModel = IE3pSpliceSitesGAModel(self.authss3pData)

	def getspecs(self):
		ret = {}
		ret["auth"] = DatasetSpec("auth", self.auth3pGAModel, self.css3pGAModel)
		ret["css"] = DatasetSpec("css", self.css3pGAModel, self.auth3pGAModel)
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

def waitForCompletion(gaModel, initPopulation, genCount, sel_provider, xover_provider, cprob, mprob):
    gaThread = GASpliceSitesThread(gaModel, initPopulation, genCount = genCount,
        crossoverProbability = cprob, mutationProbability = mprob,
        crossover_provider = xover_provider, recombine_provider = sel_provider)

    gaThread.start()
    gaThread.join()
    gaBase = gaThread.gaBase
    return gaBase


def execute(sel_provider, xover_provider, dataspecs):
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
	generationCounts = [10, 100, 1000]
	runCount = 100

	# generationSizes = [10]
	# generationCounts = [10]
	# runCount = 10

	execStat = {}
	for gensize in generationSizes:
		M = gensize
		N = 13 # 13-mers
		initPopulation = random3primeSpliceSitesPopulation(M = M, N = N)
		for gencount in generationCounts:
			for dataspec in dataspecs:
				try:
					runIdxPerfArr = []
					bestGABase = None
					bestGAFitness = -float('inf')
					for runIdx in range(runCount):
						print("EXECUTION(gensize: %d, gencount: %d, dataspec.name: %s, runIdx: %d)" % (gensize, gencount, dataspec.name, runIdx))
						gaBase = waitForCompletion(dataspec.gaModel, initPopulation, genCount=gencount, sel_provider = sel_provider, xover_provider = xover_provider, cprob = 0.7, mprob = 0.1)
						
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

	composite = Composite3pGA()
	dataspecs = composite.getspecs().values()

    # recombine_provider = lambda gaModel: lambda population: Selections.ranked(population, gaModel.fitness)
    # crossover_provider = lambda gaModel: IE3pSpliceSitesGAModel.crossover_uniform_orderbased

	selection_providers = {"rouletteWheel": lambda gaModel: lambda population: Selections.rouletteWheel(population, gaModel.fitness), 
			"ranked": lambda gaModel: lambda population: Selections.ranked(population, gaModel.fitness), 
			"tournament": lambda gaModel: lambda population: Selections.tournament(population, gaModel.fitness)}

	crossover_providers = {"crossover_1p": lambda gaModel: IE3pSpliceSitesGAModel.crossover_1p, "crossover_2p": lambda gaModel: IE3pSpliceSitesGAModel.crossover_2p, 
		"crossover_uniform": lambda gaModel: IE3pSpliceSitesGAModel.crossover_uniform, "crossover_uniform_orderbased": lambda gaModel: IE3pSpliceSitesGAModel.crossover_uniform_orderbased}

	sel = selection_providers["rouletteWheel"]
	s_name = "rouletteWheel"
	perfMap = dict()

	for x_name, xover in crossover_providers.items():
		execStatMap = execute(sel, xover, dataspecs)
		print("INITIATING EXECUTION FOR: %s" % str((s_name, x_name)))
		perfMap[(s_name, x_name)] = execStatMap
	
	xover = crossover_providers["crossover_1p"]
	x_name = "crossover_1p"
	for s_name, sel in selection_providers.items():
		print("INITIATING EXECUTION FOR: %s" % str((s_name, x_name)))
		execStatMap = execute(sel, xover, dataspecs)
		perfMap[(s_name, x_name)] = execStatMap

	return perfMap


if __name__ == '__main__':
	perfMap = main()
	stat = perfMap[('rouletteWheel', 'crossover_1p')][(10, 10, 'auth')]
	print("('rouletteWheel', 'crossover_1p') / (10, 10, 'auth'): (stat.scoreCompeteHisto: %s, stat.scoreCompeteMean: %s, stat.scoreSelfHisto: %s, stat.scoreSelfMean: %s, stat.scoreFitnessHisto: %s, stat.scoreFitnessMean: %s)" % (stat.scoreCompeteHisto, stat.scoreCompeteMean, stat.scoreSelfHisto, stat.scoreSelfMean, stat.scoreFitnessHisto, stat.scoreFitnessMean))