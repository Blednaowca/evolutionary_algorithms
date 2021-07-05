# %%
import numpy as np

np.random.seed(420)
np.set_printoptions(precision=3, sign=' ')
# %%
#constraints on position
x_min = 0
x_max = 10
#constraint on velocity
v_max = 0.5 * x_max
# acceleration coefficients
c_1 = 2
c_2 = 2

n = 4      	#number dimensions
m = 20     	#number of particles

max_iterations = 500
# %%
def alpine(positions):
	result = 1
	square_root = 1

	for position in positions:
		result *= np.sin(position)
		square_root *= position
	
	result *= np.sqrt(square_root)
	return np.abs(result)

def momentum(iteration):
	result = 0.9 - (iteration/max_iterations)
	# print(f"momentum: {result}")
	return result
# %%
class Particle:
	location = []
	velocity = []
	best_location = []
	best_location_value = 0

	def __init__(self, location, velocity):
		self.location = location
		self.velocity = velocity
		self.best_location = location
		self.best_location_value = alpine(location)
	
	def update_location(self):
		self.location = np.clip((self.location + self.velocity),x_min, x_max)
		if alpine(self.location) > self.best_location_value:
			self.best_location = self.location
			self.best_location_value = alpine(self.location)
		# return (self.best_location, self.best_location_value)

	def update_velocity(self,iteration, overall_best_location):
		r_1, r_2 = np.random.uniform(size=2)
		velocity = (c_1 * r_1 * (self.best_location - self.location) 
				  + c_2 * r_2 * (overall_best_location - self.location)
				  + momentum(iteration) * self.velocity)
		self.velocity = np.clip(velocity,-v_max, v_max)

def init_random_particle(dimensions):
	location = (np.random.random(size=dimensions) * x_max) + x_min
	velocity = (np.random.random(size=dimensions) * v_max) 
	particle = Particle(location, velocity)
	return particle

def init_particles(number_of_particles, dimensions):
	particles = []
	for i in range(number_of_particles):
		particles.append(init_random_particle(dimensions))
	return particles

def process_neighbourhood(neighbourhood):
	locations = [particle.best_location for particle in neighbourhood]
	locations_values = [alpine(location) for location in locations]

	# best_neighbourhood_location_value = np.max(locations_values)
	best_neighbourhood_location = locations[np.argmax(locations_values)]

	return best_neighbourhood_location
# %%
def main_algorithm(epsilon, printing=True):
	n = 4      	#number of dimensions
	m = 20     	#number of particles
	max_iterations = 500

	overall_best_location = np.zeros(shape=n)
	overall_best_location_value = alpine(overall_best_location)

	particles = init_particles(m, n)

	if printing:
		for particle in particles:
			print(f"initial location: {particle.location}")


	for iteration in range(max_iterations):
		for idx in range(len(particles)):
			neighbourhood = [particles[(idx+m-1)%m],
							particles[idx],
							particles[(idx+1)%m]]

			best_neighbourhood_location = process_neighbourhood(neighbourhood)
			# best_neighbourhood_location_value = alpine(best_neighbourhood_location)


			particles[idx].update_velocity(iteration, best_neighbourhood_location)
			particles[idx].update_location()

			# print((f"best: {particles[idx].best_location_value:3.3f}, " +
			# 	f"current: {alpine(particles[idx].location):3.3f}, ") +
			# 	f"velocity: {particles[idx].velocity}")

			if particles[idx].best_location_value > overall_best_location_value:
				overall_best_location = particles[idx].best_location
				overall_best_location_value = particles[idx].best_location_value
		if printing:
			print(f"Iteration {iteration + 1:3}, OBLV: {overall_best_location_value:5.5f}")
			# print()
		# print(f"{overall_best_location_value},")

	test_pos = [7.917 for i in range(n)]
	expected_result = alpine(test_pos)
	if printing:
		print(f"Expected result: {expected_result:5f}")

	percentage_acc = ((expected_result - overall_best_location_value)/
					   overall_best_location_value)

	if percentage_acc < epsilon:
		return 1
	else:
		return 0
# %%
epsilon = 0.05
main_algorithm(epsilon, printing=False)
# %%
test_pos = [7.917 for i in range(n)]
print(f"Global extremum for n = {n}: {alpine(test_pos):5f}")
# ~2.808**n
# %%
# %%
%%time
epsilon = 0.01
sum = 0
runs = 250

for _ in range(runs):
	sum += main_algorithm(epsilon, printing=False)
print(f"Algorithm has successfully  converged in {sum} out of {runs} runs")
# %%
