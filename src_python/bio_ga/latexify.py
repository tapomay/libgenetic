
def latexify(perfMap):
	'''
	('rouletteWheel', 'crossover_1p') / (10, 10, 'auth'): (stat.scoreCompeteHisto: [(1.0, 9), (0.9, 1)], stat.scoreCompeteMean: 0.99, stat.scoreSelfHisto: [(0.0, 10)], stat.scoreSelfMean: 0.0, stat.scoreFitnessHisto: [(4.3463447172747323, 1), (7.948375686832958, 1), (3.9489350437230737, 1), (0.34958729407625033, 1), (0.81301756889153154, 1), (3.2477890015590951, 1), (3.581170550257939, 2), (-0.062549850063921664, 2)], stat.scoreFitnessMean: 2.76912907127)
	Table template:
	\begin{tabular}{ |c|c||c|c| }
		\hline
		\multicolumn{2}{|c||}{col1} & \multicolumn{2}{c|}{col2}\\
		\hline
		\multicolumn{2}{|c||}{m = 10, gc = 10} & \multicolumn{2}{c|}{m = 10, gc = 20}\\
		\hline
		f($\bar{x}$) & N & f($\bar{x}$) & N \\
		\hline
		\hline
		100 & 10 & 200 & 20 \\
		\hline
		100 & 10 & 200 & 20 \\
		100 & 10 & 200 & 20 \\
		100 & 10 & 200 & 20 \\
		100 & 10 & 200 & 20 \\	
		\hline
		\multicolumn{2}{|c||}{\^{f} = 150} & \multicolumn{2}{c|}{\^{f} = 15}\\
		\hline
	\end{tabular}

	'''
	lines = []
	for (select, xover), perf1 in perfMap.items():
		lines.append("\subsection{SELECTION: %s; CROSSOVER: %s}" % (select.upper(), xover.upper()))
		dataPerfs = dict()
		for (M,N,dataname), perf2 in perf1.items():
			k = (M,N)
			if dataname not in dataPerfs:
				dataPerfs[dataname] = dict()
			dataPerfs[dataname][k] = perf2

		for dataname, dimPerfMap in dataPerfs.items():
			colPairs = len(dimPerfMap)
			colspecs = '|c|c|' * colPairs
			
			lines.append("\\begin{tabular}{ %s }" % colspecs)
			lines.append("\hline")

			tmpl0 = "\multicolumn{2}{|c||}{%s}"
			tmpl = "\multicolumn{2}{c||}{%s}"
			tmplN = "\multicolumn{2}{c|}{%s}"
			dimcols = []
			idx = 0
			for M,N in sorted(dimPerfMap):
				if idx == 0:
					tmplVal = tmpl0 % ("%s-%s" % (dataname, M))
				elif idx == len(dimPerfMap) - 1:
					tmplVal = tmplN % ("%s-%s" % (dataname, M))
				else:
					tmplVal = tmpl % ("%s-%s" % (dataname, M))
				dimcols.append(tmplVal)
				idx += 1
			lines.append(" & ".join(dimcols) + "\\\\")

			lines.append("\hline")

			dimcols = []
			idx = 0
			for M,N in sorted(dimPerfMap):
				if idx == 0:
					tmplVal = tmpl0 % ("m = %d, gc = %d" % (M,N))
				elif idx == len(dimPerfMap) - 1:
					tmplVal = tmplN % ("m = %d, gc = %d" % (M,N))
				else:
					tmplVal = tmpl % ("m = %d, gc = %d" % (M,N))
				dimcols.append(tmplVal)
				idx += 1

			lines.append(" & ".join(dimcols) + "\\\\")
			lines.append("\hline")

			lineArr = ["f($\\bar{x}$) & N" for i in range(colPairs)]
			lines.append(" & ".join(lineArr) + "\\\\")
			lines.append("\hline")
			lines.append("\hline")

			tableData = [[" " for j in range(colPairs*2)] for idx in range(10)]
			idx = 0
			for M,N in sorted(dimPerfMap):
				dimPerf = dimPerfMap[(M,N)]
				
				j = 0
				for (f,N) in dimPerf.scoreFitnessHisto:
					tableData[j][idx] = str(f)
					tableData[j][idx+1] = str(N)					
					j+= 1
					if j == 10:
						break
				idx += 2

			for tableRow in tableData:
				lines.append(" & ".join(tableRow) + "\\\\")

			lines.append("\hline")

			dimcols = []
			idx = 0
			for M,N in sorted(dimPerfMap):
				dimPerf = dimPerfMap[(M,N)]
				if idx == 0:
					tmplVal = tmpl0 % ("\\^{f} = %s" % str(dimPerf.scoreFitnessMean))
				elif idx == len(dimPerfMap) - 1:
					tmplVal = tmplN % ("\\^{f} = %s" % str(dimPerf.scoreFitnessMean))
				else:
					tmplVal = tmpl % ("\\^{f} = %s" % str(dimPerf.scoreFitnessMean))
				dimcols.append(tmplVal)
				idx += 1

			lines.append(" & ".join(dimcols) + "\\\\")

			lines.append("\hline")
			lines.append("\end{tabular}")

		# for k, dataList in dataPerfs.items():


	print("\n".join(lines))