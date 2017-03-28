import requests
import re
import httplib2
import urllib
from bs4 import BeautifulSoup

def getNextValidSibling(initialNode, node):
	nextSib = getNextSpanSibling(node)
	if nextSib is None:
		nextSib = getNextSpanSibling(initialNode.parent)
	nextToNextSib = getNextSpanSibling(nextSib)

	if nextToNextSib is not None and 'marker' in nextToNextSib['class'] and nextToNextSib.text in ['>']:
		print nextToNextSib.text
		nextSib = getNextSpanSibling(nextToNextSib)

	while nextSib is not None and nextSib.text in [')', '(', '[', ']']:
		nextSib = getNextSpanSibling(nextSib)

	if nextSib is None:
		nextSib = getNextSpanSibling(node.parent)

	return nextSib

def getNextSpanSibling(node):
	nextSib = node.nextSibling
	while nextSib is not None and nextSib.name != 'span':
		nextSib = nextSib.nextSibling
	return nextSib

def getPrevValidSibling(initialNode, node):
	prevSib = getPreviousSpanSibling(node)
	if prevSib is None:
		prevSib = getPreviousSpanSibling(initialNode.parent)
	prevToPrevSib = getPreviousSpanSibling(prevSib)

	if prevToPrevSib is not None and 'marker' in prevToPrevSib['class'] and prevToPrevSib.text in ['>']:
		print prevToPrevSib.text
		prevSib = getPreviousSpanSibling(prevToPrevSib)

	while prevSib is not None and prevSib.text in [')', '(', '[', ']']:
		prevSib = getPreviousSpanSibling(prevSib)

	if prevSib is None:
		prevSib = getPreviousSpanSibling(node.parent)

	return prevSib

def getPreviousSpanSibling(node):
	prevSib = node.previousSibling
	while prevSib is not None and prevSib.name != 'span':
		prevSib = prevSib.previousSibling

	return prevSib

baselink = "http://www.dbass.org.uk/DBASS5/"
link = "http://www.dbass.org.uk/DBASS5/viewlist.aspx"
fivePrime9Mers = []

html = requests.get(link).text
html = urllib.urlopen("dbassHtml.html").read()
soup = BeautifulSoup(html, "lxml")

for link in soup.findAll("a", {'title': 'View details for this record'}):
	nineMer = ""
	nineMer1 = ""
	hrefLink = link['href'][2:]
	print "---------------------------------------------------------------------"
	print hrefLink
	
	childHtml = requests.get(baselink + hrefLink).text
	childSoup = BeautifulSoup(childHtml, "lxml")
	seqDiv = childSoup.find("div", {'id':"PageBody_pnlSequence"})
	for marker in seqDiv.findAll("span", {"class": "marker"}, text=re.compile("/")):
		nextSib = getNextValidSibling(marker, marker)
		print nextSib

		if nextSib.text[0] in ['g', 'G'] or (len(nextSib.text) >= 2 and nextSib.text[:2] in ['gt', 'GT', 'Gt', 'gT']):
			print nextSib.text 

		textVal = nextSib.text.replace(")","")
		textVal = textVal.replace("(","")
		print textVal, "<<<---"
		if textVal.find(">") != -1:
			textVal = textVal[0:(textVal.index('>')-1)] + textVal[(textVal.index('>') + 1):]

		nineMer = textVal[:6]
			
		while len(nineMer) < 6:
			nextSib = getNextValidSibling(marker, nextSib)
			print nextSib.text
			textVal = nextSib.text.replace(")","")
			textVal = textVal.replace("(","")
			print textVal, "<<<---  ---"
			if textVal.find(">") != -1 and len(textVal) != 1:
				print textVal.index('>')
				textVal = textVal[0:(textVal.index('>')-1)] + textVal[(textVal.index('>') + 1):]
			elif textVal.find(">") != -1 and len(textVal) == 1:
				nineMer = nineMer[:(len(nineMer) - 1)]
			textVal = textVal.replace(">","")
			nineMer = nineMer + textVal[:(6 - len(nineMer))]

		prevSib = getPrevValidSibling(marker, marker)
		print prevSib

		textVal = prevSib.text.replace(")","")
		textVal = textVal.replace("(","")
		print textVal, "<<<---"
		if textVal.find(">") != -1:
			textVal = textVal[0:(textVal.index('>')-1)] + textVal[(textVal.index('>') + 1):]

		nineMer1 = textVal[-3:]
		skip = 0
		while len(nineMer1) < 3:
			print "while"
			prevSib = getPrevValidSibling(marker, prevSib)
			print prevSib.text
			
			print textVal, "<<<---  ---"
			if textVal.find(">") != -1 and len(textVal) != 1:
				textVal = prevSib.text.replace(")","")
				textVal = textVal.replace("(","")
				print textVal.index('>')
				textVal = textVal[0:(textVal.index('>')-1)] + textVal[(textVal.index('>') + 1):]
				skip = 0
			elif textVal.find(">") != -1 and len(textVal) == 1:
				prevSib = getPrevValidSibling(marker, prevSib)

			textVal = textVal.replace(">","")
			nineMer1 =  textVal[-(3 - len(nineMer1)):] + nineMer1

	nineMer = nineMer1 + nineMer
	print "9-mer", nineMer
	fivePrime9Mers.append(str(nineMer))

for fivePrime9Mer in fivePrime9Mers:
	print fivePrime9Mer

	#for marker in seqDiv.findAll("span", {"class": "marker"}, string='/('):

# import html2text

# html = open("http://www.dbass.org.uk/DBASS5/viewlist.aspx").read()
# print html2text.html2text(html)

# http = httplib2.Http()
# status, response = http.request(link)

# for link in BeautifulSoup(response, parseOnlyThese=SoupStrainer('a')):
#     if link.has_attr('href'):
#         print link['href']

# for r in res:
#     print("Text: " + r.text)
#     print("href: " + r[href])