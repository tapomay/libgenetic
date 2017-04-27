import sys
sys.path.append('src_python')
import bio_ga.report_5prime as rep
perfMap = rep.main()

# http://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions

import dill #required for lambda functions
import pickle
fname = 'perfmap_042517.pickle'
s = pickle.dumps(perfMap)
with open(fname, 'w') as f:
	f.write(s)

import bio_ga.latexify as l
summary = l.latexify(perfMap)
l.summaryTable(summary)