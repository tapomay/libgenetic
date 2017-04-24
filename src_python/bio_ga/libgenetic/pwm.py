import numpy as np

class PWM:
    '''
    dataArr: 2-D array; column count must be consistent; all symbols must be from a set
    freqMat: [A,C,G,T][1-9]
    pwmMat: freqMat /= N
    pwmLaplaceMat: (pwmMat + 1) /= (N+4)
    odds = [A:0.22, C:.28, G:.28, T:.22]
    logoddsPwmLaplaceMat: (pwmLaplaceMat / odds)
    '''
    def __init__(self, dataArr, symbolSet, symbolOddsMap):
        self._symbolSet = symbolSet
        self._cols = 0
        self._symbolOddsMap = symbolOddsMap
        if len(dataArr) > 0:
            self._cols = len(dataArr[0])
        self._pwmMatrix = self._buildMatrix(dataArr, self._symbolSet, self._cols, symbolOddsMap)

    @property
    def pwmMatrix(self):
        return self._pwmMatrix

    def _buildMatrix(self, data2D, symbolSet, colsCount, symbolOddsMap):
        '''
        Returns:
        2D matrix; (symbolCount + 1) X cols
        Each row: position weights for a symbol at each col
        Each col: positions weights at a col for each symbol
        Last row: position weights for any unrecognized symbols at each col;
                  Caller may validate for sum(retArr[-1]) == 0 to check if input had unrecognized symbols and take precautionary actions
        
        E.g.: PWM of nucleotide 9mers : 
        self._buildMatrix(data2D:List<9MER:char[9]>, symbolCount=4, cols=9):

        BASE | pos0  | pos1  | pos2  | ... | pos8  |
        --------------------------------------------
        symA | [A,0] | [A,1] | [A,2] | ... | [A,8] |
        symC | [C,0] | [C,1] | [C,2] | ... | [C,8] |
        symG | [G,0] | [G,1] | [G,2] | ... | [G,8] |
        symT | [T,0] | [T,1] | [T,2] | ... | [T,8] |
        Unkn | [X,0] | [X,1] | [X,2] | ... | [X,8] |
        --------------------------------------------
        '''
        self.freqMat = self._frequencyCounts(data2D, symbolSet, colsCount)
        self.laplace_N = len(data2D)
        self.laplacePwmMat = self._laplaceCountPWM(self.freqMat, symbolSet, self.laplace_N)
        self.logoddsMat = self._logodds(self.laplacePwmMat, symbolSet, symbolOddsMap)
        return self.logoddsMat

    def _frequencyCounts(self, dataArr2D, symbolSet, colsCount):
        retArr = np.zeros((len(symbolSet), colsCount), dtype = float) #dtype float imp. for divisions

        symIndex = list(sorted(symbolSet))

        for row in dataArr2D:
            colIdx = 0
            for sym in row:
                if sym in symIndex:
                    symRowIdx = symIndex.index(sym)
                else:
                    raise Exception("Unrecognized symbol in input: %s; symSet: %s; row: %s" % (sym, symbolSet, str(row)))

                retArr[symRowIdx][colIdx] = retArr[symRowIdx][colIdx] + 1
                colIdx += 1
        return retArr

    def _laplaceCountPWM(self, frequencyMat, symbolSet, laplace_N):
        '''
            Args:
            frequencyMat: numpy nd array
        '''
        N = laplace_N # no. of rows
        N += len(symbolSet)
        pwmMat = frequencyMat.copy()
        pwmMat = pwmMat + 1
        pwmMat = pwmMat / N # avoids zero probs
        return pwmMat

    def _logodds(self, pwmMat, symbolSet, symbolOddsMap):
        ret = pwmMat.copy()
        symIndex = list(sorted(symbolSet))
        for idx in range(len(symIndex)):
            sym = symIndex[idx]
            symOdds = symbolOddsMap[sym]
            ret[idx] = ret[idx] / symOdds # -ve logvalue => less than symProb; pos => better than symProb; zero => exactly symProb
        ret = np.log2(ret)
        return ret

    def score(self, solution):
        if len(solution) != self._cols:
            raise Exception("Column count mismatch; Expected: %s" % str(self._cols))
        ret = 0
        symIndex = list(sorted(self._symbolSet))
        for col in range(self._cols):
            sym = solution[col]
            if sym in symIndex:
                symRowIdx = symIndex.index(sym)
            else:
                raise Exception("Unrecognized symbol in input: %s; symSet: %s" % (sym, self._symbolSet))

            symScore = self.pwmMatrix[symRowIdx][col]
            ret += symScore
        return ret

    @staticmethod
    def testSelf():
        dataArr=[
            ['C', 'T', 'G', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['C', 'A', 'A', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['A', 'A', 'G', 'G', 'T', 'A', 'T', 'A', 'T'],
            ['A', 'G', 'T', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['T', 'T', 'G', 'G', 'T', 'A', 'A', 'A', 'A'],
            ['T', 'G', 'G', 'G', 'T', 'A', 'A', 'G', 'G'],
            ['C', 'A', 'G', 'G', 'T', 'G', 'A', 'G', 'T'],
            ['A', 'G', 'G', 'G', 'T', 'A', 'A', 'T', 'G'],
            ['T', 'A', 'G', 'G', 'T', 'A', 'T', 'T', 'G'],
            ['C', 'A', 'G', 'G', 'T', 'A', 'A', 'A', 'A'],
            ['A', 'A', 'G', 'G', 'T', 'G', 'T', 'G', 'T'],
            ['A', 'A', 'G', 'G', 'T', 'A', 'A', 'G', 'A'],
            ['T', 'A', 'G', 'G', 'T', 'A', 'A', 'T', 'A'],
            ['T', 'T', 'T', 'G', 'T', 'G', 'A', 'G', 'T'],
            ['C', 'A', 'G', 'G', 'T', 'A', 'T', 'A', 'C'],
            ['T', 'C', 'T', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['G', 'A', 'G', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['A', 'A', 'G', 'G', 'T', 'A', 'A', 'A', 'G'],
            ['C', 'A', 'G', 'G', 'T', 'A', 'A', 'G', 'A'],
            ['A', 'C', 'A', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['A', 'T', 'G', 'G', 'T', 'A', 'A', 'G', 'G']
        ]
        symbolSet = set(['A', 'C', 'G', 'T'])
        symbolOddsMap = {'A':0.28, 'C': 0.22, 'G': 0.22, 'T': 0.28}
        pwm = PWM(dataArr, symbolSet, symbolOddsMap)
        print(pwm.pwmMatrix)

        testSolution = "CTGGTAAGT"
        testScore = pwm.score(testSolution)
        print(testScore)
        return pwm

    @staticmethod
    def testSelf_3prime():
        dataArr=[
            ['C', 'T', 'G', 'G', 'T', 'A', 'A', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['C', 'A', 'A', 'G', 'T', 'A', 'A', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['A', 'A', 'G', 'G', 'T', 'A', 'T', 'A', 'T', 'A', 'A', 'G', 'T'],
            ['A', 'G', 'T', 'G', 'T', 'A', 'A', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['T', 'T', 'G', 'G', 'T', 'A', 'A', 'A', 'A', 'A', 'A', 'G', 'T'],
            ['T', 'G', 'G', 'G', 'T', 'A', 'A', 'G', 'G', 'A', 'A', 'G', 'T'],
            ['C', 'A', 'G', 'G', 'T', 'G', 'A', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['A', 'G', 'G', 'G', 'T', 'A', 'A', 'T', 'G', 'A', 'A', 'G', 'T'],
            ['T', 'A', 'G', 'G', 'T', 'A', 'T', 'T', 'G', 'A', 'A', 'G', 'T'],
            ['C', 'A', 'G', 'G', 'T', 'A', 'A', 'A', 'A', 'A', 'A', 'G', 'T'],
            ['A', 'A', 'G', 'G', 'T', 'G', 'T', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['A', 'A', 'G', 'G', 'T', 'A', 'A', 'G', 'A', 'A', 'A', 'G', 'T'],
            ['T', 'A', 'G', 'G', 'T', 'A', 'A', 'T', 'A', 'A', 'A', 'G', 'T'],
            ['T', 'T', 'T', 'G', 'T', 'G', 'A', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['C', 'A', 'G', 'G', 'T', 'A', 'T', 'A', 'C', 'A', 'A', 'G', 'T'],
            ['T', 'C', 'T', 'G', 'T', 'A', 'A', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['G', 'A', 'G', 'G', 'T', 'A', 'A', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['A', 'A', 'G', 'G', 'T', 'A', 'A', 'A', 'G', 'A', 'A', 'G', 'T'],
            ['C', 'A', 'G', 'G', 'T', 'A', 'A', 'G', 'A', 'A', 'A', 'G', 'T'],
            ['A', 'C', 'A', 'G', 'T', 'A', 'A', 'G', 'T', 'A', 'A', 'G', 'T'],
            ['A', 'T', 'G', 'G', 'T', 'A', 'A', 'G', 'G', 'A', 'A', 'G', 'T']
        ]
        symbolSet = set(['A', 'C', 'G', 'T'])
        symbolOddsMap = {'A':0.28, 'C': 0.22, 'G': 0.22, 'T': 0.28}
        pwm = PWM(dataArr, symbolSet, symbolOddsMap)
        print(pwm.pwmMatrix)

        testSolution = "CTGGTAAGTAAGT"
        testScore = pwm.score(testSolution)
        print(testScore)
        return pwm

if __name__ == '__main__':
    PWM.testSelf()