import random
from evol import Population, Evolution
import numpy as np
from collections import OrderedDict
#import matplotlib.pyplot as plt

#np.set_printoptions(suppress=True)

# def recast_to_ordered(in_dict):
#   out_dict = OrderedDict()
#   for k in in_dict.keys():
#     out_dict[k] = in_dict[k]

#   return out_dict

def generate_uniform_distribution(from_value, to_value, popsize):
  if type(from_value) == int and type(to_value) == int:
    return np.random.randint(from_value, to_value, popsize).tolist()
    #TODO: cast  it to list of floats...
  else:
    return np.random.uniform(from_value, to_value, popsize).tolist()
  
def choose_randomly(choices, popsize):
    return np.random.choice(choices,popsize).tolist()

# def from_to_start(start_values, popsize):
#   attributes = []
#   for k in start_values.keys():
    
#     try:
#       assert len(start_values[k])==2
#       assert all(type(x) == int or type(x) == float for x in start_values[k])
#     except Exception as e:
#       print("This function can only be used with pure numeric ranges.")
#       raise e

#     attributes.append(generate_uniform_distribution(start_values[k][0],start_values[k][1],popsize))
  
#   attributes = np.array(attributes).T#.tolist()

#   return list(map(tuple,attributes))


def from_to_and_choice_start(start_values, popsize):
  attributes = {}

  for k in start_values.keys():
    
    if len(start_values[k])==2 and all(type(x) == int or type(x) == float for x in start_values[k]):
      attributes[k] = generate_uniform_distribution(start_values[k][0],start_values[k][1],popsize)
    else:
      attributes[k] = choose_randomly(start_values[k],popsize)
  
  individuals = []
  for count in range(popsize):
    individual = {}
    
    for k in attributes.keys():
      individual[k] = attributes[k][count]
    
    individuals.append(individual)

  return individuals


# def dummy_model_function(attributes):

#   error = abs(int(attributes[0])-13) 
#   error += (abs(float(attributes[1])-0.2))*100.0
#   error += abs(float(attributes[2])+8.0)
#   if attributes[3]!="ketto":
#     error+=30

#  all_scores.append(abs(error))
#  return abs(error)

def pick_random_parents(pop):
  """
  This is how we are going to select parents from the population
  """
  mom = random.choice(pop)
  dad = random.choice(pop)
  return mom, dad

def make_child(mom, dad):
  keys = list(mom.keys())
  cutpoint = random.choice(range(len(keys)))
  mom_keys = keys[:cutpoint]
  dad_keys = keys[cutpoint:]

  child = {}

  for k in mom_keys:
    child[k] = mom[k]

  for k in dad_keys:
    child[k] = dad[k]

  return child

def mut(chromosome, sigma=1, start_values={}):

  new = {}

  #start_values = list(start_values.values())

  for k in chromosome.keys():

    if type(chromosome[k]) == float: #np.float64:
      new[k] = chromosome[k] + ((random.random()-0.5) * (start_values[k][1]-start_values[k][0])*sigma)
      if new[k]>start_values[k][1]:
        new[k]=start_values[k][1]
      if new[k]<start_values[k][0]:
        new[k]=start_values[k][0]
      
    elif type(chromosome[k]) == int: #np.int64:
      beginning = start_values[k][0]
      end = start_values[k][1]
      full_range = end-beginning
      half = int((end-beginning)/2)

      element = chromosome[k] + int((random.randint(beginning,end)-half)*sigma)
      assert type(element) == int #np.int64
      new[k] = element

      if new[k]>start_values[k][1]:
        new[k]=start_values[k][1]
      if new[k]<start_values[k][0]:
        new[k]=start_values[k][0]

    elif type(chromosome[k]) == str:
      prob = random.random()
      if prob>1-sigma:
        element = random.choice(start_values[k])
        assert type(element)== str
        new[k] = element
      else:
        element = chromosome[k]
        assert type(element)==str
        new[k] = element
    else:
      print(type(chromosome[k]))
      raise 

  return new


def evolve(start_values, model_function, popsize, iterations, pop_memory=[], fitness_memory=[], maximize=False, survior_fraction=0.5, mutation_sigma=0.1):
  
  pop = Population(chromosomes=from_to_and_choice_start(start_values,popsize),
                 eval_function=model_function, maximize=maximize)

  evol = (Evolution()
       .survive(fraction=survior_fraction, luck=True)
       .breed(parent_picker=pick_random_parents, combiner=make_child)
       .mutate(mut, sigma=mutation_sigma, start_values=start_values))

  pop = pop.evolve(evol, n=iterations)
  pop.evaluate()

  best_params = pop.documented_best.chromosome
  best_score = pop.documented_best.fitness

  #print(pop_memory)
  print("Number of evaluated genomes in pop_memory:",len(pop_memory),"\n\n")

  return best_params, best_score


#---------------
# popsize = 20
# start_values = {}
# start_values["alma"]=[1,100] #13
# start_values["korte"]=[0.0,1.0] #0.2
# start_values["barack"]=[-10.0,0.0] #-8.0
# start_values["vadkorte"]=["egy","ketto","harom"] #"ketto"

# solution, score = evolve(start_values,dummy_model_function,20,10)

# print("Ground truth:")
# print("alma: 13, korte: 0.2, barack: -8.0, vadkorte: ketto")

# print(solution)
# print(score)

#TODO: filter duplicates???