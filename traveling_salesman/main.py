# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)

from data import *
from plot import *
# %%
x = [0, 2, 6,   7,   15,  12, 14, 9.5, 7.5, 0.5]
y = [1, 3, 5, 2.5, -0.5, 3.5, 10, 7.5,   9, 10]
cities = pd.DataFrame(zip(range(len(x)), x, y), columns=['n', 'x', 'y'])
cities
# %%
def calculate_distance(cities: pd.DataFrame, city_a: int, city_b: int):
	distance = np.sqrt((cities.x[city_a] - cities.x[city_b])**2 +
					   (cities.y[city_a] - cities.y[city_b])**2)
	return distance

# %%
def calculate_total_distance(arr: np.array, cities: pd.DataFrame=cities):
	total_distance = 0
	for i in range(len(arr) - 1):
		total_distance += calculate_distance(cities, arr[i], arr[i + 1])
	total_distance += calculate_distance(cities, arr[-1], arr[0])
	return total_distance

# %%
def calculate_fitness(distances):
	fitness = []
	max_dist = np.max(distances)
	for i in range(len(distances)):
		fitness.append(max_dist - distances[i])
	return fitness
# %%
def calculate_distances(population):
	distances = []
	for i in range(len(population)):
		distances.append(calculate_total_distance(population[i]))
	return distances
# %%
def parents_selection(population, distances, number_of_parents):
	fitness = calculate_fitness(distances)
	total_fitness = sum(fitness)
	# print(f"Total fitness = {total_fitness}")

	parents = []

	for i in range(number_of_parents):
		fitness_sum = 0
		random_number = np.random.uniform(high=total_fitness)
		for i in range(len(fitness)):
			fitness_sum += fitness[i]
			if fitness_sum >= random_number:
				parents.append(population[i])
				break
		# print(fitness_sum)

	return parents
# %%
def offspring_creation(parents, number_of_offspring):
	offsprings = []
	for _ in range(number_of_offspring):
		ai, bi = np.random.choice(len(parents), 2)
		parent_a = parents[ai]
		parent_b = parents[bi]

		#crossover
		offspring = np.array([None for _ in range(len(parent_a))])
		
		i = 0
		while True:
			# i = parent_b.index(parent_a[i])
			i = np.where(parent_b == parent_a[i])[0]
			offspring[i] = parent_a[i][0]

			if i == 0:
				break

		for i in range(len(offspring)):
			if offspring[i] is None:
				offspring[i] = parent_b[i]
				

		offsprings.append(offspring.astype(int))
	return offsprings
# %%
def mutation(offsprings, mutation_probability=0.2):
	for offspring in offsprings:
		if np.random.uniform() < mutation_probability:
			idx_1, idx_2 = np.random.choice(offspring, 2)
			offspring[idx_1], offspring[idx_2] = offspring[idx_2], offspring[idx_1]
# %%
def get_n_smallest(distances: list, n):
	return sorted(range(len(distances)), key=lambda k: distances[k])[:n]

def adjust_population(parents, offsprings, pop_size=250):
	whole_pop = [*parents, *offsprings]

	distances = calculate_distances(whole_pop)
	indices = get_n_smallest(distances, pop_size)

	new_pop = [whole_pop[i] for i in indices]

	return new_pop
# %%
def print_n_smallest(population, n):
	distances = calculate_distances(population)
	n_smallest = get_n_smallest(distances, n)

	for i in n_smallest:
		print(f"{distances[i]:.9f}, {np.hstack([population[i], population[i][0]])}")
	return population[n_smallest[0]]
# %%
pop_size = 250 # parameter P
Tmax = 100 # max epochs
n = 0.8
n_offsprings = int(n * pop_size)
n_parents = int(n * pop_size)
#%%
N = len(cities)
plot_cities(cities)
# %%
init_pop = [np.random.permutation(N) for _ in range(pop_size)]
# %%
population = init_pop.copy()
# %%
number_of_epochs = 0 # just for printing, not necessary
# %%
best_pop = []
for i in range(Tmax):
	print(number_of_epochs)
	distances = calculate_distances(population)
	# fit = calculate_fitness(distances)
	parents = parents_selection(population, distances, n_parents)
	offsprings = offspring_creation(parents, n_offsprings)
	mutation(offsprings)
	population = adjust_population(parents, offsprings, pop_size=pop_size)

	# prints 5 smallest distances in new population
	best_pop = print_n_smallest(population, 5)

	number_of_epochs += 1
	print('=' * 25 + '\n')
# %%
def tsp_algorithm(population, pop_size, n, mutation_probability):
	n_offsprings = int(n * pop_size)
	n_parents = int(n * pop_size)
	number_of_epochs = 0

	for i in range(Tmax):
		if number_of_epochs % 10 == 0:
			print(number_of_epochs)
		distances = calculate_distances(population)
		parents = parents_selection(population, distances, n_parents)
		offsprings = offspring_creation(parents, n_offsprings)
		mutation(offsprings, mutation_probability=mutation_probability)
		population = adjust_population(parents, offsprings, pop_size=pop_size)
		distances = calculate_distances(population)
		number_of_epochs += 1
	return distances[0]
# %%
plot_path(cities, best_pop)
# %%
best_pop
# %%
pops = [500]
ns = [0.7]
mutation_probabilities = [0.1, 0.3, 0.5]
N = len(cities)
Tmax = 100
# %%
results = []
# %%
for pop_size in pops:
	for n in ns:
		for mutation_probability in mutation_probabilities:
			total_distance = 0

			for _ in range(10):
				population = [np.random.permutation(N) for _ in range(pop_size)]

				partial_distance = tsp_algorithm(population, pop_size, n, mutation_probability)
				print(f"partial_distance = {partial_distance}")
				total_distance += partial_distance
			
			mean_total_distance = total_distance / 10
			print(f"Mean of minimum total distance = {mean_total_distance}")
			results.append((pop_size, n, mutation_probability, mean_total_distance))

	


# %%
calculate_total_distance(best_pop)
# %%
results

# %%
