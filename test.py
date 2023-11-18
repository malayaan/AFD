# Importing pandas
import pandas as pd

# Reload the data
file_path = 'chienloup.csv'
data = pd.read_csv(file_path, sep=';')

# Selecting only numerical columns for calculations
numeric_columns = data.select_dtypes(include=[float, int]).columns

# Splitting the data into subgroups based on the 'GENRE' column
groups = data.groupby('GENRE')

# Calculating variance-covariance matrix for each subgroup
var_cov_matrices = {group: df[numeric_columns].cov() for group, df in groups}

# Calculating mean vectors for each subgroup, using only numerical columns
mean_vectors_complete = {group: df[numeric_columns].mean() for group, df in groups}

# Calculating the overall mean vector g
overall_mean = data[numeric_columns].mean()

# Total number of samples n
n = data.shape[0]

# Calculating B𝑝×𝑝
B = sum([groups.get_group(group).shape[0] * (mean_vectors_complete[group] - overall_mean).values.reshape(-1, 1) @
         (mean_vectors_complete[group] - overall_mean).values.reshape(1, -1) for group in groups.groups]) / n

# Calculating W𝑝×𝑝 (Within-class variance matrix)
W = sum([groups.get_group(group).shape[0] * var_cov_matrices[group] for group in groups.groups]) / n

print(B)

print(W)

V= B+W

print(V)