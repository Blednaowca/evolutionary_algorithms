# %%
import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(0)
# %%
data_raw = np.genfromtxt('data/model1.txt')
X, Y = data_raw[:,0], data_raw[:,1]
# %% plotting the data
plt.scatter(X, Y)
# %% optimized function
def f(x):
	a, b, c = x
	i = np.linspace(X[0], X[-1], X.shape[0])
	return a*(i**2 - b*np.cos(c*np.pi*i))
# %% error function
def E(x):
	return np.sum((Y - f(x))**2)/len(X)
# %% generate solution
def generate_solution(T, x0):
	return x0 + np.random.normal(0, T, x0.shape)
# %% cooling functions
def lin_cooling(T, n, Tmin=1e-4):
	if T > 10:
		return T - n
	else:
		return max(T - 0.02*n, Tmin)

def exp_cooling(T, alpha):
	return alpha * T

def inv_cooling(T, beta):
	return T / (1 + beta * T)
# %% annealing loop
T0 = 1e6
T = T0
epsilon = 1e-6
iters = 0

n = 0.1
alpha = 0.9999
beta = 0.001
q = inv_cooling #choose cooling type

x0 = np.random.uniform(-10, 10, 3)
x = x0

history = []

while T > 1e-3 or abs(Ex - Ex0) > epsilon:

	x = generate_solution(T, x0)
	Ex = E(x)
	Ex0 = E(x0)

	print(f"{x}, {Ex}, {T}")

	if Ex < Ex0: 
		x0 = x
		
	else:
		r = np.random.uniform()
		if r < np.e**((Ex0 - Ex) / T):
			x0 = x
	
	history.append((T, Ex, x))

	# print(T)

	
	
	T = q(T, beta)
	iters += 1

# %%
epsilon = 1e-6

alpha = 0.9999
beta = 0.0001

q = inv_cooling
# [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8] for n in [0.1, 0.01, 0.001]
for n in [0.0001]:
	for T0 in [1e4, 1e5]:
		# print(f"T0 = {T0}")
		# print(f"n = {n}")
		T = T0

		x0 = np.random.uniform(-10, 10, 3)
		x = x0

		history = []
		while T > 1e-3 or abs(Ex - Ex0) > epsilon:
			# print(f"T = {T}")
			x = generate_solution(T, x0)
			Ex = E(x)
			Ex0 = E(x0)

			if Ex < Ex0: 
				x0 = x
				
			else:
				r = np.random.uniform()
				if r < np.e**((Ex0 - Ex) / T):
					x0 = x
			
			history.append([T, Ex, x])
			
			T = q(T, n)

		history = np.array(history, dtype=object)
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
		ax1.scatter(X, f(x))
		ax2.plot(history[:,1])
		ax2.set_yscale('log') 
		plt.show()
		print(f"T0 = {T0}, n = {n}")
		print(x, Ex)
		print()
# %%
print(f"Number of iterations: {iters}")
history = np.array(history)
plt.scatter(X, f(x))
# %%
plt.plot(history[:,1])
plt.yscale('log')
# %%
x
# %%
Ex
# %%
1e1
# %%
