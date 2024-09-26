import json

### Get result ###
#Load JSON data from file
with open("./ael_results/pops/population_generation_1.json") as file:
    data = json.load(file)


#Print each individual in the population

for individual in data:
    #print(individual)
    results = individual
    code = results['code']
    algorithm = results['algorithm']
    gap = results['objective']
    
    #code2file(code)
    
    print("### algorithm: \n",algorithm)
    print("### code: \n",code)
    print("### Average gap is : \n",gap)
    input()

