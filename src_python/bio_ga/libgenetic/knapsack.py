from libgenetic import EvolutionBasic, Selections, Crossovers, Mutations, Generation, GABase
from libgenetic import randomPopulation, f_to_intarr
import numpy as np
import random

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
        p_file = 'data/knapsack/p01_p.txt'
        w_file = 'data/knapsack/p01_w.txt'
        c_file = 'data/knapsack/p01_c.txt'
        s_file = 'data/knapsack/p01_s.txt'
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
            raise Exception("dimension mismatch: %s" % solutionArr)
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
    knapsack = GAKnapsack.loadProblem(p_file = 'data/knapsack/p01_p.txt', w_file = 'data/knapsack/p01_w.txt', 
        c_file = 'data/knapsack/p01_c.txt', s_file = 'data/knapsack/p01_s.txt')
    main_inst(knapsack, M, N, GEN_COUNT=200)

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