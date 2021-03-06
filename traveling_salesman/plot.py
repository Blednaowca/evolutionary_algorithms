import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot(cities, T, bestAnt):
	Tmax = 0
	for row in T:
		for t in row:
			if t > Tmax:
				Tmax = t

	for i, city in enumerate(cities):
		plt.scatter(city.x, city.y, c='royalblue')
		for anotherCity in cities[0:i] + cities[i+1:len(cities)]:
			lw = 0.5 + 3*T[city['n']][anotherCity['n']]/Tmax
			plt.plot([city.x, anotherCity.x], [city.x, anotherCity.x], c='gray', lw=lw, zorder=0)

	for i in range(len(cities)):
		n1 = bestAnt['path'][i]
		n2 = bestAnt['path'][i+1]
		plt.plot([cities[n1].x, cities[n2].x], [cities[n1].x, cities[n2].x], c='orange', zorder=0)
		
	for city in cities:
		plt.annotate(city.n, (city.x + 0.1, city.y + 0.1))
	plt.show()

def plot_cities(cities: pd.DataFrame):
	for i, city in cities.iterrows():
		plt.scatter(city.x, city.y, c='royalblue')
		for _, other_city in cities[cities.index != i].iterrows():
			plt.plot([city.x, other_city.x], [city.y, other_city.y], lw=0.2, c='gray', zorder=0)
		
	for _, city in cities.iterrows():
		plt.annotate(int(city.n), (city.x + 0.1, city.y + 0.1))
	plt.show()

def plot_path(cities: pd.DataFrame, pop: np.ndarray):
	for i, city in cities.iterrows():
		plt.scatter(city.x, city.y, c='royalblue')
		for _, other_city in cities[cities.index != i].iterrows():
			plt.plot([city.x, other_city.x], [city.y, other_city.y], lw=0.2, c='gray', zorder=0)
		
	for _, city in cities.iterrows():
		plt.annotate(int(city.n), (city.x + 0.1, city.y + 0.1))		
	
	for i in range(len(pop)):
		n1 = pop[i]
		try: n2 = pop[i+1]
		except: n2 = pop[0]
		
		plt.plot([cities.x[n1], cities.x[n2]], [cities.y[n1], cities.y[n2]], c='orange', zorder=0)
	plt.show()