import extract_css as em
import pickle
parser3prime = None
with open('css_3prime.tsv', 'w') as f:
    def outputCollector(outLineArr):
        for outLine in outLineArr:
            f.write("%s\n" % outLine.strip().upper())
            f.flush()
    parser3prime = em.Dbass3Parser(manualCheck = False, executor = None, outputCollector = outputCollector)
    ret = parser3prime.parseAll(pageCount = 1500)
    print "Total 9-mers found: %d" % len(ret)

with open('dbass3Parser.pickle', 'w') as fparser:
    s = pickle.dumps(parser3prime)
    fparser.write(s)

loadParser3prime = None
with open('dbass3Parser.pickle', 'r') as f:
    s = f.read()
    outputCollector = lambda:None
    loadParser3prime = pickle.loads(s)


    # with futures.ProcessPoolExecutor() as executor:
    #     p = em.Dbass3Parser(manualCheck = False, executor = None, outputCollector = outputCollector)
    #     ret = p.parseAll(pageCount = 1500)
    #     print "Total 9-mers found: %d" % len(ret)

    #     with open('Dbass5Parser.pickle', 'w') as fparser:
    #         s = pickle.dumps(p)
    #         fparser.write(p)
