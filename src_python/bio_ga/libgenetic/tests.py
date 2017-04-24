from libgenetic import EvolutionBasic, Selections, Crossovers, Mutations, Generation, GABase
from pwm import PWM
import numpy as np


TEST_SS_DATA = ['', '', '', '']
FITNESS_GLOBAL = 1
'''
Selections
'''
def test_rouletteWheel():

	def fitnessFunction(chromo):
		global FITNESS_GLOBAL
		FITNESS_GLOBAL = FITNESS_GLOBAL + 1
		return FITNESS_GLOBAL

	population = [['A', 'C', 'G', 'T', 'A', 'C', 'G', 'T', 'A'], ['A', 'A', 'C', 'G', 'T', 'A', 'C', 'G', 'T'],
		['T', 'A', 'A', 'C', 'G', 'T', 'A', 'C', 'G'], ['G', 'T', 'A', 'A', 'C', 'G', 'T', 'A', 'C']]
	ret = Selections.rouletteWheel(population, fitnessFunction)
	print(ret)

def test_rankedSelection():

	def fitnessFunction(chromo):
		global FITNESS_GLOBAL
		FITNESS_GLOBAL = FITNESS_GLOBAL + 1
		return FITNESS_GLOBAL

	population = [['A', 'C', 'G', 'T', 'A', 'C', 'G', 'T', 'A'], ['A', 'A', 'C', 'G', 'T', 'A', 'C', 'G', 'T'],
		['T', 'A', 'A', 'C', 'G', 'T', 'A', 'C', 'G'], ['G', 'T', 'A', 'A', 'C', 'G', 'T', 'A', 'C']]
	ret = Selections.ranked(population, fitnessFunction)
	print(ret)

def test_tournamentSelection():

	def fitnessFunction(chromo):
		global FITNESS_GLOBAL
		FITNESS_GLOBAL = FITNESS_GLOBAL + 1
		return FITNESS_GLOBAL

	population = [['A', 'C', 'G', 'T', 'A', 'C', 'G', 'T', 'A'], ['A', 'A', 'C', 'G', 'T', 'A', 'C', 'G', 'T'],
		['T', 'A', 'A', 'C', 'G', 'T', 'A', 'C', 'G'], ['G', 'T', 'A', 'A', 'C', 'G', 'T', 'A', 'C']]
	ret = Selections.tournament(population, fitnessFunction)
	print(ret)


print("ROULETTE")
test_rouletteWheel()
print("RANKED")
test_rankedSelection()
print("TOURNAMENT")
test_tournamentSelection()