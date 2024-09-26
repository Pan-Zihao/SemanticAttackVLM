import json

### Get result ###
#Load JSON data from file
with open("../ael_results/history/pop_0_3_crossover_.json") as file:
    data = json.load(file)


#Print each individual in the population

for individual in data:
    print(individual)
    results = data[individual]
    code = results['code']
    algorithm = results['algorithm']
    gap = results['objective']
    
    #code2file(code)
    
    print("### algorithm: \n",algorithm)
    print("### code: \n",code)
    print("### Average gap is : \n",gap)
    input()

