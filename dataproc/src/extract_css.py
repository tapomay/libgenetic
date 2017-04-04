import requests
import re
import httplib2
import urllib
from bs4 import BeautifulSoup
import sys
import concurrent.futures as futures
import time
import pickle

LEFT_PARANTHESIS = '('
RIGHT_PARANTHESIS = ')'
GREATER_THAN = '>'
FORWARD_SLASH = '/'
WAIT_TIME = 3

THREE_PRIME_SS_INDEX_MIN_THRESHOLD = 10 # 13 - (2:AG + 1:post-slice)

def log(logline, *args):
	if args:
		print(logline % args)
	else:
		print(logline)


class Dbass5Parser:

	def __init__(self, manualCheck = False, executor = None, outputCollector = None):
		self._manualCheck = manualCheck
		self._executor = None
		self._outputCollector = None
		if executor:
			self._executor = executor
		if outputCollector:
			self._outputCollector = outputCollector
		self._htmldataArr = dict()
		self._cssArr = dict()

	def parseAll(self, urlTemplate = "http://www.dbass.org.uk/DBASS5/viewsplicesite.aspx?id=%s", 
		pageCount = 1000):
		ret = []
		futureObjects = []
		for idx in range(pageCount):
			link = urlTemplate % idx
			try:
				if self._executor:
					future = self._executor.submit(self.linkParse, link, self.page_parse)
					futureObjects.append(future)
				else:
					linkCssArr = self.linkParse(urlStr = link, parser = self.page_parse)
					if linkCssArr:
						ret = ret + linkCssArr
						cssData = [data[1] for data in linkCssArr]
						print("PAGE: %s: CSS_ARR: %s" % (link, str(cssData)))
						if self._outputCollector:
							self._outputCollector(cssData)

				if len(futureObjects) > 0:
					for f in futureObjects:
						linkCssArr = f.result()
						if linkCssArr:
							ret = ret + linkCssArr
							cssData = [data[1] for data in linkCssArr]
							print("PAGE: %s: CSS_ARR: %s" % (link, str(cssData)))
							if self._outputCollector:
								self._outputCollector(cssData)

			except BaseException as e:
				print(e)
			sys.stdout.flush()

		return ret

	def linkParse(self, urlStr, parser):
		time.sleep(WAIT_TIME)
		resp = requests.get(urlStr)
		if resp.status_code == 200:
			log("Found CSS page: %s", urlStr)
			self._htmldataArr[urlStr] = resp
			ret =  parser(resp.text)
			self._cssArr[urlStr] = ret
			return ret
		else:
			raise Exception("Invalid url: %s; resp.status_code: %s" % (urlStr, str(resp.status_code)))

	def page_parse(self, pageHtmlData):
		soup = BeautifulSoup(pageHtmlData, "lxml")
		seqDiv = soup.find("div", {'id':"PageBody_pnlSequence"})
		# log("Parsing sequence: %s", seqDiv)
		
		whitelist = ["intron", "exon", "pseudoexon"]

		doParse = 1
		if self._manualCheck:
			doParse = input("Press 1 to parse, 0 to skip: ")
		
		if doParse == 0:
			return None

		markersArr = seqDiv.findAll("span")

		markersTextArr = [m.text for m in markersArr if m.get('class')[0] != 'sequenceChunk']
		sequenceStr = "".join(markersTextArr)

		# find and resolve mutation marker
		mutatedSequenceStr = self._resolveMutation(sequenceStr)

		# find splice marker
		spliceIndices = [match.start() for match in re.finditer(FORWARD_SLASH, mutatedSequenceStr)]
		
		if len(spliceIndices) < 1:
			raise Exception("No splicing marker: %s" % sequenceStr)

		ret = []
		for spliceSite in spliceIndices:
			if mutatedSequenceStr[spliceSite+1: spliceSite+3] in ['gt', 'GT', 'Gt', 'gT']:
				css9mer = mutatedSequenceStr[spliceSite-3: spliceSite] + mutatedSequenceStr[spliceSite+1: spliceSite+7]
				pre9mer = mutatedSequenceStr[0:spliceSite-3]
				post9mer = mutatedSequenceStr[spliceSite+7:]
				ret.append((pre9mer, css9mer, post9mer))
				log("Detected CSS at %d" % spliceSite)

			else:
				print("Skipped splice index: %d, mut_seq: %s" %(spliceSite, mutatedSequenceStr))

		if not ret:
			raise Exception("Failed to detect any splice index: %s" % sequenceStr)

		return ret


	def _resolveMutation(self, sequenceStr):
		# find and resolve mutation marker

		if LEFT_PARANTHESIS not in sequenceStr:
			raise Exception("No mutation - no left paranthesis: %s" % sequenceStr)
		lp_idx = sequenceStr.index(LEFT_PARANTHESIS)
		
		if RIGHT_PARANTHESIS not in sequenceStr:
			raise Exception("Malformed mutation - no right paranthesis: %s" % sequenceStr)
		rp_idx = sequenceStr.index(RIGHT_PARANTHESIS)
		if rp_idx - lp_idx != 4:
			raise Exception("Malformed mutation marker: %s" % sequenceStr)

		mutation = sequenceStr[lp_idx:rp_idx+1]
		updatedNucleotide = mutation[3]
		log("Mutation detected: %s to %s", mutation[1], mutation[3])
		mutatedSequenceStr = sequenceStr.replace(mutation, updatedNucleotide)
		return mutatedSequenceStr

class Dbass3Parser(Dbass5Parser):

	def parseAll(self, urlTemplate = "http://www.dbass.org.uk/DBASS3/viewsplicesite.aspx?id=%s", 
		pageCount = 1000):
		return Dbass5Parser.parseAll(self, urlTemplate = urlTemplate, pageCount = pageCount)

	def page_parse(self, pageHtmlData):
		soup = BeautifulSoup(pageHtmlData, "lxml")
		seqDiv = soup.find("div", {'id':"PageBody_pnlSequence"})
		# log("Parsing sequence: %s", seqDiv)
		
		whitelist = ["intron", "exon", "pseudoexon"]

		doParse = 1
		if self._manualCheck:
			doParse = input("Press 1 to parse, 0 to skip: ")
		
		if doParse == 0:
			return None

		markersArr = seqDiv.findAll("span")

		markersTextArr = [m.text for m in markersArr if m.get('class')[0] != 'sequenceChunk']
		sequenceStr = "".join(markersTextArr)

		# find and resolve mutation marker
		mutatedSequenceStr = self._resolveMutation(sequenceStr)

		# find splice marker
		spliceIndices = [match.start() for match in re.finditer(FORWARD_SLASH, mutatedSequenceStr)]
		
		if len(spliceIndices) == 0:
			raise Exception("No splicing marker: %s" % sequenceStr)

		ret = []

		for spliceSite in spliceIndices:
			# avoid indexoutofbounds and check for YYYYNAG/... consensus
			if spliceSite > THREE_PRIME_SS_INDEX_MIN_THRESHOLD and \
				mutatedSequenceStr[spliceSite - 2:spliceSite] in ['ag', 'aG', 'Ag', 'AG'] :
					# create 13-mer for 3 prime splice site
					# (10 bases before AG ++ AG) ++ one base after AG
					css13mer = mutatedSequenceStr[spliceSite-12: spliceSite] + mutatedSequenceStr[spliceSite+1]
					pre13mer = mutatedSequenceStr[0:spliceSite-12]
					post13mer = mutatedSequenceStr[spliceSite+2:]
					log("Detected CSS at %d" % spliceSite)
					ret.append((pre13mer, css13mer, post13mer))
			else:
				print("Skipped splice index: %d, mut_seq: %s" %(spliceSite, mutatedSequenceStr))

		if not ret:
			raise Exception("Failed to detect any splice index: %s" % sequenceStr)

		return ret

def main():
	import extract_css as em
	# p.linkParse("http://www.dbass.org.uk/DBASS5/viewsplicesite.aspx?id=722", parser = p.pageParse)
	with open('css.tsv', 'w') as f:
		
		def outputCollector(outLine):
			f.write("%s\n" % outLine.strip().upper())
			f.flush()

		with futures.ProcessPoolExecutor() as executor:
			p = em.Dbass5Parser(manualCheck = False, executor = None, outputCollector = outputCollector)
			ret = p.parseAll(pageCount = 1500)
			# print "\n".join([r[1] for r in ret])
			print "Total 9-mers found: %d" % len(ret)

			with open('Dbass5Parser.pickle', 'w') as fparser:
				s = pickle.dumps(p)
				fparser.write(p)


if __name__ == '__main__':
	main()