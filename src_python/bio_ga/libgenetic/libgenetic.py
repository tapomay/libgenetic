
'''
Object Oriented Class Design for generic Genetic Algorithm library

Entity: String | Object
EntityIndex: Map<String, int>

states: List<String>
cardinality = len(states)

T: Array<bool|int>
Solution: <T>
Population -> List<Solution>, M:int, N:int
matingPool: Population

Generation -> Population, index, Map<Generation, FitnessScore>
Evolution: List<Generation>

Fitness: Function<Solution> : Score
Penalty: Function<Solution> : Score
FitnessFunction: Fitness | Function<Fitness, Penalty> : Score
Score: Integer

Objective: Function<Solution>: Integer
Constraint: Function<Solution, Condition> : Boolean

Condition: binrayCondition | (op, Condition, Condition)
binrayCondition : <Solution>: Boolean

'''
import random
import numpy as np

class Generation:
    
    def __init__(self, population, genIndex = 0, fitness = None):
        self._population = population
        self._genIndex = genIndex
        self._fitness = fitness
        self._bestFitness = None
        self._bestSolution = None
        self._matingPool = None
        self._xover = None

    def __str__(self):
        return "Gen: %s, BestFitness: %s, BestSolution: %s, Fitness: %s, Pop: %s" %(self._genIndex, self._bestFitness, self._bestSolution, self._fitness, self._population)

class EvolutionBasic:

    def __init__(self, select, crossover, mutate, crossoverProbability = 0.1, mutationProbability = 0.1):
        '''
            select: F(): 
            crossover: F(): 
            mutate: F(): 
        '''
        self._select = select
        self._crossover = crossover
        self._mutate = mutate
        self._crossoverProbability = crossoverProbability
        self._mutationProbability = mutationProbability

    def _recombine_pool(self, matingPool):
        ret = matingPool[:]
        indices = range(len(matingPool))
        # print("XOVER: matingPool: %s" % matingPool)
        def pickSolution(pool, idxArr):
            pickIdx = random.randint(0, len(idxArr) - 1)
            # print("XOVER: pick: %d" % pickIdx)
            # print("XOVER: indices: %s" % idxArr)
            solIdx = idxArr[pickIdx]
            sol = pool[solIdx]
            idxArr.remove(solIdx)
            return (sol, solIdx)

        totalPairs = len(indices) / 2
        randomXoverCount = int(self._crossoverProbability * totalPairs)
        if randomXoverCount == 0:
            randomXoverCount = 1
        print("randomXoverCount: %d" % randomXoverCount)
        for i in range(randomXoverCount):
            sol1, sol1Idx = pickSolution(matingPool, indices)
            sol2, sol2Idx = pickSolution(matingPool, indices)
            # print("XOVER: parent1: %s" % sol1)
            # print("XOVER: parent2: %s" % sol2)
            (child1, child2) = self._crossover(sol1, sol2) # @TODO: if crossoverProbability
            # print("XOVER: child1: %s" % child1)
            # print("XOVER: child2: %s" % child2)
            ret[sol1Idx] = child1
            ret[sol2Idx] = child2

        return ret
    
    def _mutate_pool(self, recombinedPool):
        ret = recombinedPool[:]
        indices = range(len(recombinedPool))
        def pickIdx(idxArr):
            idx = random.randint(0, len(idxArr) - 1)
            solIdx = idxArr[idx]
            idxArr.remove(solIdx)
            return solIdx

        mutateCount = int(self._mutationProbability * len(indices))
        if mutateCount == 0:
            mutateCount = 1
        print("mutateCount: %d" % mutateCount)
        for i in range(mutateCount):
            solIdx = pickIdx(indices)
            ret[solIdx] = self._mutate(recombinedPool[solIdx])
        return ret

    def evolve(self, gen):
        matingPool = self._select(gen._population)
        pool_recombined = self._recombine_pool(matingPool)
        pool_next = self._mutate_pool(pool_recombined)
        retGen  = Generation(pool_next, genIndex = gen._genIndex + 1)
        retGen._matingPool = matingPool
        retGen._xover = pool_recombined
        return retGen

class EvolutionRankOrdered:
    pass

class GABase:

    def __init__(self, evolution, gen0, fitnessFunction):
        self._generations = [gen0]
        self._initGen = gen0
        self._evolution = evolution
        self._currentGen = gen0
        self._fitnessFunction = fitnessFunction

    def _stepGeneration(self):
        newGen = self._evolution.evolve(self._currentGen)
        self._generations.append(newGen)
        self._currentGen = newGen
        return newGen

    def _isStable(self): # @TODO: check fitness growth
        return True

    def execute(self, maxGens = 10):
        for i in range(maxGens):
            if self._isStable():
                newGen = self._stepGeneration()
                newGen._fitness = [self._fitnessFunction(solution) for solution in newGen._population]
                newGen._bestFitness = max(newGen._fitness)
                newGen._bestSolution = newGen._population[newGen._fitness.index(newGen._bestFitness)]
                print(newGen)
                self._generations.append(newGen)

def f_to_intarr(fname):
    ret = []
    with open(fname) as f:
        lines_arr = f.readlines()
        for line in lines_arr:
            line = line.strip()
            ret.append(int(line))
    return ret

class Crossovers:

    '''
    All crossover methods should satisfy the following signature:
     fn_name(parent1, parent2, site = None): returns (child1, child2)

    '''
    @staticmethod
    def one_point(parent1, parent2, site = None):
        parent1Len = len(parent1)
        if len(parent1) != len(parent2):
            raise Exception("Incompatible lengths: parent1: %d vz parent2: %d" % (len(parent1), len(parent2)))

        if not site:
            site = random.randint(1, parent1Len - 2)
        print("Selected Xover site: %d" % site)
        p1_sub = parent1[site:]
        p2_sub = parent2[site:]
        child1 = parent1[0:site] + p2_sub
        child2 = parent2[0:site] + p1_sub
        return (child1, child2)

    @staticmethod
    def two_point(parent1, parent2, site = None):
        parent1Len = len(parent1)
        if len(parent1) != len(parent2):
            raise Exception("Incompatible lengths: parent1: %d vz parent2: %d" % (len(parent1), len(parent2)))

        if not site:
            site = pick_random_site(parent1Len)
        print("Selected Xover site: %d" % site)
        p1_sub = parent1[site:]
        p2_sub = parent2[site:]
        child1 = parent1[0:site] + p2_sub
        child2 = parent2[0:site] + p1_sub
        return (child1, child2)

    @staticmethod
    def pick_random_site(range = 9, negativeSites = []):
        if range < 3:
            raise Exception("Invalid usage: range should be atleast 3")
        site = random.randint(1, range - 2)
        if negativeSites:
            while site in negativeSites:
                site = random.randint(1, range - 2)
        return site

class Mutations:
    @staticmethod
    def bool_flip(solution):
        site = random.randint(0, len(solution) - 1)
        print("Selected mutation site: %d" % site)
        val = solution[site]
        val = abs(1 - val)
        ret = solution[:]
        ret[site] = val
        return ret

    @staticmethod
    def provided_flip(solution, flipProvider, negativeSites = []):
        site = random.randint(0, len(solution) - 1)
        if negativeSites:
            while site in negativeSites:
                site = random.randint(0, len(solution) - 1)
        print("Selected mutation site: %d" % site)
        val = solution[site]
        newVal = flipProvider(val)
        ret = solution[:]
        ret[site] = newVal
        return ret

class Selections:

    @staticmethod
    def rouletteWheel(population, fitnessFunction):
        '''
            Arguments:
            population: Array<Solution>: size=N
            fitnessFunction: function(Solution): Double

            Returns:
            Array<Solution>: size=N
            randomly selected solutions from the given population based on a Roulette wheel 
            the roulette wheel is scaled to the worst fitness
            with probability distribution on the fitness values returned by fitnessFunction
        '''

        # compute fitness of each solution in population
        fitnessArr = []
        for solution in population:
            fitness = fitnessFunction(solution)
            fitnessArr.append(fitness)
        minFitness = min(fitnessArr)
        # print fitnessArr
        fitnessArr = np.divide(fitnessArr, float(minFitness)) #scaling fitness values
        # print fitnessArr
        popFitness = float(sum(fitnessArr))

        # normalize fitnessArr
        normFitnessArr = map(lambda x: x / popFitness, fitnessArr)
        # print "normFit: %s" % str(normFitnessArr)

        # compute cumulative normalized fitness values
        # note that this is a strictly increasing array
        cumulative = 0
        normFitnessCumulativeArr = []
        for norm in normFitnessArr:
            cumulative = cumulative + norm
            normFitnessCumulativeArr.append(cumulative)

        # print "normCumFit: %s" % str(normFitnessCumulativeArr)

        # Exactly N times (where N = len(population)),
        # generate a random integer between [0,1]
        # find the index in normalized cumulative fitness array where the cumulative fitness exceeds the random number
        # select and append the solution from population at the previous index
        ret = []
        for i in range(len(population)):
            spin = random.random()
            # print "spin: %f" % spin
            for j in range(1, len(normFitnessCumulativeArr)):
                if spin < normFitnessCumulativeArr[j]:
                    # print "Select: %d, pop: %s, normCumFitness: %f" % (j-1, population[j-1], normFitnessCumulativeArr[j-1])
                    ret.append(population[j-1])
                    break
        return ret

    @staticmethod
    def ranked(population, fitnessFunction):
        '''
            Arguments:
            population: Array<Solution>: size=N
            fitnessFunction: function(Solution): Double

            Returns:
            Array<Solution>: size=N
        '''

        # compute fitness of each solution in population
        fitnessTupleArr = []
        for solution in population:
            fitness = fitnessFunction(solution)
            fitnessTupleArr.append((solution, fitness))
        # print fitnessTupleArr
        fitnessTupleArrSorted = sorted(fitnessTupleArr, reverse=True, key = lambda x:x[1]) # sort by fitness
        # print fitnessTupleArrSorted


        popFitness = float(sum([x[1] for x in fitnessTupleArrSorted]))

        # normalize fitnessArr
        normFitnessTupArr = map(lambda x: (x[0], x[1] / popFitness), fitnessTupleArrSorted)
        # print "normFit: %s" % str(normFitnessTupArr)

        # compute cumulative normalized fitness values
        # note that this is a strictly increasing array
        cumulative = 0
        normFitnessCumulativeArr = []
        for norm in normFitnessTupArr:
            cumulative = cumulative + norm[1]
            normFitnessCumulativeArr.append(cumulative)

        # print "normCumFit: %s" % str(normFitnessCumulativeArr)

        # Exactly N times (where N = len(population)),
        # generate a random integer between [0,1]
        # find the index in normalized cumulative fitness array where the cumulative fitness exceeds the random number
        # select and append the solution from population at the previous index
        ret = []
        for i in range(len(population)):
            spin = random.random()
            # print "spin: %f" % spin
            for j in range(1, len(normFitnessCumulativeArr)):
                if spin < normFitnessCumulativeArr[j]:
                    # print "Select: %d, pop: %s, normCumFitness: %f" % (j-1, population[j-1], normFitnessCumulativeArr[j-1])
                    ret.append(fitnessTupleArrSorted[j-1][0])
                    break
        return ret

    @staticmethod
    def tournament(population, fitnessFunction, s=2):
        '''
            Arguments:
            population: Array<Solution>: size=N
            fitnessFunction: function(Solution): Double

            Returns:
            Array<Solution>: size=N
        '''
        # for each m:
        #   select s individuals : generate s random indicies in range(m)
        #   add the fittest individual to selection
        ret = []
        m = len(population)
        for idx in range(m):
            compete = random.sample(population, s)
            competeFitness = [fitnessFunction(individual) for individual in compete]
            winnerIdx = np.argmax(competeFitness)
            winner = compete[winnerIdx]
            ret.append(winner)
        return ret

class GAKnapsack:

    @staticmethod
    def loadProblem(p_file, w_file, c_file, s_file):
        profits = f_to_intarr(p_file)
        weights = f_to_intarr(w_file)
        capacity = f_to_intarr(c_file)[0]
        solution = f_to_intarr(s_file)
        return GAKnapsack(profits, weights, capacity, solution)

    @staticmethod
    def loadTestProblem():
        p_file = 'data/p01_p.txt'
        w_file = 'data/p01_w.txt'
        c_file = 'data/p01_c.txt'
        s_file = 'data/p01_s.txt'
        profits = f_to_intarr(p_file)
        weights = f_to_intarr(w_file)
        capacity = f_to_intarr(c_file)[0]
        solution = f_to_intarr(s_file)
        return GAKnapsack(profits, weights, capacity, solution)

    def __init__(self, profitsArr, weightsArr, capacity, solution):
        self._profits = profitsArr
        self._weights = weightsArr
        self._capacity = capacity
        self._solution = solution
        self._N = len(self._profits)
        self._maxProfit = max(self._profits)

    def fitness(self, solutionArr):
        if len(solutionArr) != self._N:
            raise Exception("dimension mismatch")
        indices = range(self._N)
        profits = [self._profits[i] for i in indices if solutionArr[i] == 1]
        profits = sum(profits)
        weights = [self._weights[i] for i in indices if solutionArr[i] == 1]
        weights = sum(weights)
        penalty = 0
        if weights > self._capacity:
            penalty = self._N * self._maxProfit
        ret = profits - penalty
        return ret

    def weight(self, solutionArr):
        if len(solutionArr) != self._N:
            raise Exception("dimension mismatch")
        indices = range(self._N)
        weights = [self._weights[i] for i in indices if solutionArr[i] == 1]
        weights = sum(weights)
        return weights

    def profit(self, solutionArr):
        if len(solutionArr) != self._N:
            raise Exception("dimension mismatch")
        indices = range(self._N)
        profits = [self._profits[i] for i in indices if solutionArr[i] == 1]
        profits = sum(profits)
        return profits

    def __str__(self):
        return "P: %s\nW: %s\nC: %s\nS: %s" % (str(self._profits), str(self._weights), str(self._capacity), str(self._solution))

def randomPopulation(M, N, cardinality=2):
    randomgen = lambda: random.randint(0, cardinality - 1)
    ret = []
    for i in range(M):
        sol = []
        for j in range(N):
            bit = randomgen()
            sol.append(bit)
        ret.append(sol)
    return ret

def main():
    M = 10
    N = 10
    initPopulation = randomPopulation(M, N, 2)
    knapsack = GAKnapsack.loadProblem(p_file = 'data/p01_p.txt', w_file = 'data/p01_w.txt', 
        c_file = 'data/p01_c.txt', s_file = 'data/p01_s.txt')
    main_inst(knapsack, M, N, GEN_COUNT=100)

def main_inst(knapsack, M, N, GEN_COUNT=10):
    initPopulation = randomPopulation(M, N, 2)
    print(knapsack)

    gen0 = Generation(initPopulation)
    recombine = lambda population: Selections.rouletteWheel(population, knapsack.fitness)
    evolution = EvolutionBasic(select = recombine, crossover= Crossovers.two_point, mutate = Mutations.bool_flip)
    gaBase = GABase(evolution, gen0, knapsack.fitness)
    gaBase.execute(maxGens=GEN_COUNT)
    genFitness = [gen._bestFitness for gen in gaBase._generations]
    bestFitness_all = max(genFitness)
    bestGenArr = filter(lambda g: g._bestFitness == bestFitness_all, gaBase._generations)
    print ""
    for bestGen in bestGenArr:
        print("BEST: %s, profit: %s, weight: %s" % (bestGen, knapsack.profit(bestGen._bestSolution), knapsack.weight(bestGen._bestSolution)))
    print "SOLUTION: %s, solutionFitness: %s, solutionProfit: %s, solutionWeight: %s" %(knapsack._solution, 
        knapsack.fitness(knapsack._solution), knapsack.profit(knapsack._solution), knapsack.weight(knapsack._solution))

def main_test2():
    M = 10
    N = 10
    initPopulation = randomPopulation(M, N, 2)
    knapsack = GAKnapsack.loadProblem()
    print(knapsack)

    gen0 = Generation(initPopulation)
    recombine = lambda population: Selections.rouletteWheel(initPopulation, knapsack.fitness)
    evolution = EvolutionBasic(select = recombine, crossover = Crossovers.two_point, mutate = Mutations.bool_flip)
    gaBase = GABase(evolution, gen0, knapsack.fitness)
    newGen = gaBase._stepGeneration()
    gen0FitnessArr = [knapsack.fitness(solution) for solution in newGen._population]
    gen0Fitness = max(gen0FitnessArr)
    gen0._fitness = gen0FitnessArr
    print(gen0)
    fitnessArr = [knapsack.fitness(solution) for solution in newGen._population]
    newGenFitness = max(fitnessArr)
    newGen._fitness = fitnessArr
    print(newGen)

def main_test():
    M = 10
    N = 10
    initPopulation = randomPopulation(M, N, 2)
    print(initPopulation)
    knapsack = GAKnapsack.loadProblem()
    print(knapsack)
    print(knapsack.fitness(knapsack._solution))
    (c1, c2) = Crossovers.two_point(initPopulation[0], initPopulation[1])
    print("%s ==> %s" %(initPopulation[0], c1))
    print("%s ==> %s" %(initPopulation[1], c2))

    print("%s ==> %s" %(c1, Mutations.bool_flip(c1)))
    gen0 = Generation(initPopulation)

    rouletteResult = Selections.rouletteWheel(initPopulation, knapsack.fitness)
    print(rouletteResult)

    recombine = lambda population: Selections.rouletteWheel(initPopulation, knapsack.fitness)
    evolution = EvolutionBasic(select = recombine, crossover = Crossovers.two_point, mutate = Mutations.bool_flip)
    xoveredPool = evolution._recombine_pool(rouletteResult)
    print (xoveredPool)

    # gaBase = GABase(evolution, gen0)
    # newgen = gaBase._stepGeneration()
    # print(newGen)

if __name__ == '__main__':
    main()