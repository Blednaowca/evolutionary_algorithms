import re
import pandas as pd

def get_data(path):
	lines = []
	x, y = [], []
	with open(path) as file:
		for line in file:
			lines.append(re.findall('\d+', line))

	x = [int(number) for number in lines[1]]
	y = [int(number) for number in lines[2]]
	return pd.DataFrame(zip(range(len(x)), x, y), columns=['n', 'x', 'y'])
