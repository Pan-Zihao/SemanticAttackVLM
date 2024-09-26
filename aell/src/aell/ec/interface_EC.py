import numpy as np
import time
from .evolution import Evolution
from .selection import parent_selection


class InterfaceEC:
    def __init__(self, pop_size, m, api_endpoint, api_key, llm_model, debug_mode, interface_eval, object=False, **kwargs):
        # -------------------- RZ: use local LLM --------------------
        assert 'use_local_llm' in kwargs
        assert 'url' in kwargs
        # -----------------------------------------------------------

        # LLM settings
        self.pop_size = pop_size
        self.interface_eval = interface_eval
        self.evol = Evolution(api_endpoint, api_key, llm_model, debug_mode, object=object, **kwargs)
        self.m = m
        
    #def code2file(self,code):
        #with open("./ael_alg.py", "w") as file:
        # Write the code to the file
        #    file.write(code)
        #print("code",code)
        #pass
    #    return

    def code2file(self, text, filename="caption.txt"):
        with open(filename, "w", encoding="utf-8") as file:
            file.write(text)

    def add2pop(self,population,offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True
    
    def check_duplicate(self,population,code):
        for ind in population:
            if code == ind['code']:
                return True
        return False

    # def population_management(self,pop):
    #     # Delete the worst individual
    #     pop_new = heapq.nsmallest(self.pop_size, pop, key=lambda x: x['objective'])
    #     return pop_new
    
    # def parent_selection(self,pop,m):
    #     ranks = [i for i in range(len(pop))]
    #     probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
    #     parents = random.choices(pop, weights=probs, k=m)
    #     return parents

    def population_generation(self):
        population = []
        n_create = 0
        while (len(population)!=self.pop_size):
            n_create += 1
            _, individual = self.get_algorithm(population,'i1')
            self.add2pop(population,individual)
            print(f"algorithm with fitness {individual['objective']} has been designed")
            
        print("Initiliazation finished! create "+str(n_create)+" times ")

        return population
    
    def population_generation_seed(self,seeds):
        population = []

        for seed in seeds:
            seed_alg = {
                'algorithm': seed['algorithm'],
                'code': seed['code'],
                'objective': None,
                'other_inf': None
            }
            self.code2file(seed_alg['code'])
            try:
                fitness = self.interface_eval.evaluate()
            except Exception as e:
                fitness = None
                print("Error in seed algorithm")
                exit()
            fitness = np.array(fitness)
            print(fitness)
            seed_alg['objective'] = np.round(fitness, 5)
            population.append(seed_alg)

        print("Initiliazation finished! Get "+str(len(seeds))+" seed algorithms")

        return population
    

    def _get_alg(self,pop,operator):
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
        if operator == "i1":
            parents = None
            [offspring['code'],offspring['algorithm']] =  self.evol.i1()            
        elif operator == "e1":
            parents = parent_selection(pop,self.m)
            [offspring['code'],offspring['algorithm']] = self.evol.e1(parents)
        elif operator == "e2":
            parents = parent_selection(pop,self.m)
            [offspring['code'],offspring['algorithm']] = self.evol.e2(parents) 
        elif operator == "m1":
            parents = parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']] = self.evol.m1(parents[0])   
        elif operator == "m2":
            parents = parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']] = self.evol.m2(parents[0]) 
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n") 

        return parents, offspring


    def get_algorithm(self,pop,operator):
        
        p,offspring = self._get_alg(pop,operator)
        while self.check_duplicate(pop,offspring['code']):
            print("duplicated code, wait 1 second and retrying ... ")
            time.sleep(1)
            p,offspring = self._get_alg(pop,operator)
        self.code2file(offspring['code'])
        try:
            fitness= self.interface_eval.evaluate()
        except:
            fitness = None
        offspring['objective'] =  fitness
        #offspring['other_inf'] =  first_gap
        while (fitness == None):
            print("warning! error code, retrying ... ")
            p,offspring = self._get_alg(pop,operator)
            while self.check_duplicate(pop,offspring['code']):
                print("duplicated code, wait 1 second and retrying ... ")
                time.sleep(1)
                p,offspring = self._get_alg(pop,operator)
            self.code2file(offspring['code'])
            try:
                fitness= self.interface_eval.evaluate()
            except:
                fitness = None
            offspring['objective'] =  fitness
            #offspring['other_inf'] =  first_gap
        offspring['objective'] = np.round(offspring['objective'],5) 
        #offspring['other_inf'] = np.round(offspring['other_inf'],3)
        return p,offspring
