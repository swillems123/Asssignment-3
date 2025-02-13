import pandas as pd
import numpy as np

# File paths
petal_path = r"C:\Users\sethw\Documents\se\Petal_Data.csv"
sepal_path = r"C:\Users\sethw\Documents\se\Sepal_Data.csv"

# Load the CSV files
df_petal = pd.read_csv(petal_path)
df_sepal = pd.read_csv(sepal_path)

# Merge the two datasets on 'sample_id' and 'species'
df = pd.merge(df_petal, df_sepal, on=["sample_id", "species"])

# The combined DataFrame now contains:
# sample_id, species, petal_length, petal_width, sepal_length, sepal_width
print("Combined DataFrame:")
print(df.head())

# Define measurement columns
measurement_cols = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']

# 1. Correlation between each measurement
correlation_matrix = df[measurement_cols].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# 2. Average of each measurement
averages = df[measurement_cols].mean()
print("\nAverages:")
print(averages)

# 3. Median of each measurement
medians = df[measurement_cols].median()
print("\nMedians:")
print(medians)

# 4. Standard Deviation of each measurement
std_devs = df[measurement_cols].std()
print("\nStandard Deviations:")
print(std_devs)

# 5. Compare species by their mean measurements
species_means = df.groupby('species')[measurement_cols].mean()
print("\nSpecies Mean Measurements:")
print(species_means)

def euclidean_distance(a, b):
    """Calculate the Euclidean distance between two vectors."""
    return np.sqrt(np.sum((a - b) ** 2))

# Calculate distances between species based on their mean measurements
species = species_means.index.tolist()
distances = {}
for i in range(len(species)):
    for j in range(i + 1, len(species)):
        sp1, sp2 = species[i], species[j]
        distance = euclidean_distance(species_means.loc[sp1].values, species_means.loc[sp2].values)
        distances[f"{sp1} vs {sp2}"] = distance

print("\nEuclidean Distances Between Species Means:")
for pair, distance in distances.items():
    print(f"{pair}: {distance}")