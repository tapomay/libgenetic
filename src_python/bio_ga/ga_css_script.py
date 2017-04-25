from tabulate import tabulate
from collections import OrderedDict
# import ga_css as ga
import ga_css_3prime as ga

generationSizes = [10, 50]
# genCounts = [10, 20, 30]
genCounts = [10, 20, 30, 40, 50, 100, 500, 1000]

statsMap = OrderedDict()
for s in generationSizes:
    for c in genCounts:
        genStats = ga.main(generationSize=s, genCount = c)
        statKey = "gensize: %d, gencount: %d" % (s, c)
        statsMap[statKey] = genStats

for genKey, genStats in statsMap.items():
    print("\n%s" % genKey)
    print(tabulate(genStats, headers='firstrow'))


# c = 0.7
# m = 0.1

'''
Implement 13-mers from 3' splice sites
Computation notes:
For 5' splice sites:
    4^7 = 16384 9mers
    With gen_size = 10:
        search space = 16384 ninemers
        or 16384(Combine10) different sets of 9mers

    For GA:
        With gen_count = 100, gen_size = 10
        search space explored = 100 * 10 = 1000 9mers
        or 100 different sets of 9mers
    Hence, search space explored = 1000 / 16384 ~~ 6.25%

For 3' splice sites:
    13-mers due to consensus: -----YYYYYYNAG|----
    In above computation:
        4^7 changes to 4^11 = 4194304
    This strengthens the case for GA vz exhaustive search
    Data: http://www.sci.unisannio.it/docenti/rampone/
    
'''