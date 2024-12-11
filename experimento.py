import numpy as np

def rastrigin_function(x):
    return 10 * len(x) + sum([(xi**2 - 10 * np.cos(2 * np.pi * xi)) for xi in x])

def initialize_population(pop_size, dim, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, (pop_size, dim))

def evaluate_population(population, objective_function):
    return np.array([objective_function(ind) for ind in population])

def select_best(population, fitness, num_best):
    indices = np.argsort(fitness)[:num_best]
    return population[indices], fitness[indices]

def clone_population(population, fitness, clone_factor):
    clones = []
    for i, individual in enumerate(population):
        num_clones = max(1, int(clone_factor / (1 + fitness[i])))
        clones.extend([individual] * num_clones)
    return np.array(clones)

def hypermutation(clones, mutation_rate, lower_bound, upper_bound):
    mutated_clones = []
    for clone in clones:
        mutation = np.random.uniform(-mutation_rate, mutation_rate, size=clone.shape)
        mutated_clone = clone + mutation
        mutated_clone = np.clip(mutated_clone, lower_bound, upper_bound)
        mutated_clones.append(mutated_clone)
    return np.array(mutated_clones)

def replace_worst(population, fitness, new_individuals, new_fitness):
    if len(new_individuals) == 0:
        print("Nenhum novo indivíduo gerado nesta geração.")
        return population, fitness
    combined_population = np.vstack((population, new_individuals))
    combined_fitness = np.hstack((fitness, new_fitness))
    indices = np.argsort(combined_fitness)[:len(population)]
    return combined_population[indices], combined_fitness[indices]

def ais(objective_function, dim, pop_size=50, num_generations=100, 
        clone_factor=10, mutation_rate=0.1, num_best=10, lower_bound=-5.12, upper_bound=5.12):
    population = initialize_population(pop_size, dim, lower_bound, upper_bound)
    fitness = evaluate_population(population, objective_function)

    for generation in range(num_generations):
        best_individuals, best_fitness = select_best(population, fitness, num_best)
        clones = clone_population(best_individuals, best_fitness, clone_factor)
        print(f"Geração {generation + 1}: {len(clones)} clones gerados.")
        
        mutated_clones = hypermutation(clones, mutation_rate, lower_bound, upper_bound)
        if len(mutated_clones) == 0:
            print(f"Geração {generation + 1}: Nenhum clone mutado.")
        
        mutated_fitness = evaluate_population(mutated_clones, objective_function)
        population, fitness = replace_worst(population, fitness, mutated_clones, mutated_fitness)
        print(f"Geração {generation + 1}: Melhor fitness = {min(fitness)}")

    best_index = np.argmin(fitness)
    return population[best_index], fitness[best_index]

if __name__ == "__main__":
    print("Iniciando o algoritmo AIS...\n")
    best_solution, best_fitness = ais(rastrigin_function, dim=10, num_generations=50)
    print("\nMelhor solução encontrada:", best_solution)
    print("Fitness da melhor solução:", best_fitness)
