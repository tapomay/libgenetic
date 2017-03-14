import ga_css as ga
from ga_css import GASpliceSites, GASpliceSitesThread, randomSpliceSitesPopulation
cssFile = 'data/splicesite_data/CrypticSpliceSite.tsv'
authssFile = 'data/splicesite_data/EI_true_9.tsv'
cssGA = GASpliceSites.load_data_tsv(cssFile)
authssGA = GASpliceSites.load_data_tsv(authssFile)
cssGASpliceSites = GASpliceSites(cssGA)
authGASpliceSites = GASpliceSites(authssGA)
M = 10
N = 9 # 9-mers
initPopulation = randomSpliceSitesPopulation(M, N)
authThread = GASpliceSitesThread(authGASpliceSites, initPopulation, genCount = 10)
cssThread = GASpliceSitesThread(cssGASpliceSites, initPopulation, genCount = 10)
cssThread.start()
cssThread.join()
print(ga.check_match(bestGen._population, cssGA))