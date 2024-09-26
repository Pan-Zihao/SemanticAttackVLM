import heapq

def population_management(pop,size):
    if size > len(pop):
        size = len(pop)
    # Delete the worst individual
    pop_new = heapq.nsmallest(size, pop, key=lambda x: x['objective'])
    return pop_new