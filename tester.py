import inspect

def dummy_model(attributes):
	global pop_memory
	global fitness_memory

	# Target values for this dummy:
	# {"alma: 13, korte: 0.2, barack: -8.0, vadkorte: "ketto"}
	
	#print(globals(),"\n\n")
	error = abs(int(attributes["alma"])-13) 
	error += (abs(float(attributes["korte"])-0.2))*100.0
	error += abs(float(attributes["barack"])+8.0)
	if str(attributes["vadkorte"])!="ketto":
		error+=10
	
	pop_memory.append(attributes)
	fitness_memory.append(error)

	return abs(error)



#----<mockup model function>----

#def model_function(attributes):
	"""Define a function that runs the desired training loop and
	returns one scalar value, preferrably the best validation loss for the model

	Accept an attributes dict, which contains the hyperparams, 
	keys are the same as for the parameter_ranges."""

	#use the maximize=True / False in evol accordingly.
	#default: False 
	#best_validation_loss = ... #inf or -inf, or such...


	#return best_validation_loss

#----<mockup model function>----



#-----<evolution>------------

from evolution import evolve

popsize = 20
generations = 10

parameter_ranges = {"alma":[1,100],
					"korte":[0.0,1.0],
					"barack":[-10.0,0.0],
					"vadkorte":["egy","ketto","harom"]}
pop_memory=[]
fitness_memory=[]

solution, score = evolve(parameter_ranges, dummy_model, popsize, generations, pop_memory, fitness_memory)

#-----</evolution>------------



print("Ground truth:")
print("alma: 13, korte: 0.2, barack: -8.0, vadkorte: ketto")
print(solution)
print(score)
