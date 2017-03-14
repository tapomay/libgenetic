from tabulate import tabulate
from collections import OrderedDict
import ga_css as ga

generationSizes = [10, 20, 30, 40, 50, 100]
genCounts = [10, 20, 30]

statsMap = OrderedDict()
for s in generationSizes:
    for c in genCounts:
        genStats = ga.main(generationSize=s, genCount = c)
        statKey = "gensize: %d, gencount: %d" % (s, c)
        statsMap[statKey] = genStats

for genKey, genStats in statsMap.items():
    print("\n%s" % genKey)
    print(tabulate(genStats, headers='firstrow'))