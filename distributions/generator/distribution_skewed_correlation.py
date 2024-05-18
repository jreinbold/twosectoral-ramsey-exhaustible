import csv
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap



# =============================================================================
# define parameters
# =============================================================================

num_entries = 10000
mean = 1

# =============================================================================
# generate random values for productivity and initial wealth
# =============================================================================

def generate_skewed_values(mean, shape, scale, size):
    values = np.random.gamma(shape, scale, size)
    # Adjust values to ensure they have 2 decimal places and mean of 1
    values = np.round(values, decimals=2)
    values *= mean / np.mean(values)
    return values

shape = 2  # Shape parameter for skewness, adjust as needed
scale = 0.2  # Scale parameter, adjust as needed

min_wage_factor = 0.3
productivity = generate_skewed_values(mean-min_wage_factor, shape, scale, num_entries)
productivity = productivity+min_wage_factor

min_wealth_factor = 0.25
initial_wealth = generate_skewed_values(mean-min_wealth_factor, shape, scale, num_entries)
initial_wealth = initial_wealth+min_wealth_factor

print(productivity.mean()) # to make sure mean equals 1
print(initial_wealth.mean())  # to make sure mean equals 1


data = {'productivity': productivity,'initial_wealth': initial_wealth}
df = pd.DataFrame(data)

# Create correlation via sorting and shuffling
df['productivity'] = df['productivity'].sort_values().reset_index(drop=True)
df['initial_wealth'] = df['initial_wealth'].sort_values().reset_index(drop=True)
df.loc[num_entries*0.1:num_entries*0.9, 'productivity'] = df.loc[num_entries*0.1:num_entries*0.9, 'productivity'].sample(frac=1).values

# Visualize existing distribtuion
#df = pd.read_csv("path")

# Save newly generated distirbution
#df.to_csv('skewed_correlation_distribution.csv', index=False)


# =============================================================================
# Single plot distributions
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].hist(df['initial_wealth'], bins=20, alpha=0.7)
axes[0].set_xlabel('initial_wealth')
axes[0].set_ylabel('Frequency')

axes[1].hist(df['productivity'], bins=20, alpha=0.7)
axes[1].set_xlabel('productivity')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# =============================================================================
# calculate quintiles
# =============================================================================

num_rows = len(df)
quintiles_size = num_rows // 5

df = df.sort_values(by=['productivity'])
df.reset_index(drop=True, inplace=True)
df['productivity_quintile'] = np.repeat(np.arange(1, 6), quintiles_size)

# Calculate terciles for 'wealth'
df = df.sort_values(by=['initial_wealth'])
df.reset_index(drop=True, inplace=True)
df['initial_wealth_quintile'] = np.repeat(np.arange(1, 6), quintiles_size)


# =============================================================================
# Single plot quintiles
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

productivity_quintile_sums = df.groupby('productivity_quintile')['productivity'].sum()
productivity_quintile_sums.plot(kind='bar', ax=axes[0])
axes[0].set_title('productivity_quintile')
axes[0].tick_params(axis='x', rotation=0)

initial_wealth_quintile_sums = df.groupby('initial_wealth_quintile')['initial_wealth'].sum()
initial_wealth_quintile_sums.plot(kind='bar', ax=axes[1])
axes[1].set_title('initial_wealth')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()

# =============================================================================
# calculate joint distribution frequencies
# =============================================================================

ct = pd.crosstab(df['initial_wealth_quintile'], df['productivity_quintile'])

frequencies = np.array(ct)
frequencies = frequencies/df.shape[0] # from absolute counts to percentages

frequencies = frequencies[:, [4,3,2,1,0]]  # Reorder the y columns

# =============================================================================
# Single plot joint distribution frequencies
# =============================================================================

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

xlabels = np.array(['5th', '4th', '3rd', '2nd', '1st']) # productivity labels
ylabels = np.array(['1st', '2nd', '3rd', '4th', '5th']) # wealth labels

xpos, ypos = np.arange(xlabels.shape[0]), np.arange(ylabels.shape[0])
xposM, yposM = np.meshgrid(xpos, ypos)

zpos = 0

dx = dy = 0.5 * np.ones_like(zpos)
dz = frequencies.ravel()


ax.bar3d(xposM.ravel(), yposM.ravel(), zpos, dx, dy, dz, zsort='average')
ax.set_xticks(xpos)
ax.set_xticklabels(xlabels)
ax.set_yticks(ypos)
ax.set_yticklabels(ylabels)
ax.set_xlabel('productivity')
ax.set_ylabel('initial_wealth')
ax.set_zlabel('Frequency in %')

plt.show()

# =============================================================================
# Combined plots
# =============================================================================

# Set up a figure twice as tall as it is wide
fig = plt.figure(figsize=plt.figaspect(0.33))

# First subplot
ax = fig.add_subplot(1, 3, 1)

ax.hist(df['productivity'], bins=50, alpha=0.7)
ax.set_xlabel('Productivity')
ax.set_ylabel('Frequency')
ax.set_title("(a) Productivity distribution", y=-0.3)


# First subplot
ax = fig.add_subplot(1, 3, 2)

ax.hist(df['initial_wealth'], bins=50, alpha=0.7)
ax.set_xlabel('Initial wealth')
ax.set_title("(b) Initial wealth distribution", y=-0.3)


# Third 3d subplot
ax = fig.add_subplot(1, 3, 3, projection='3d')

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

surf = ax.bar3d(xposM.ravel(), yposM.ravel(), zpos, dx, dy, dz, zsort='average')
ax.set_xticks(xpos)
ax.set_xticklabels(xlabels)
ax.set_yticks(ypos)
ax.set_yticklabels(ylabels)
ax.set_xlabel('Productivity')
ax.set_ylabel('Initial wealth')
ax.set_zlabel('Frequency in %')
ax.set_title("(c) Joint distribution" , y=-0.41)

# Save combined plot
# plt.savefig('distribution_skewed_correlation.png', bbox_inches='tight',pad_inches = 0.3,dpi = 200)
plt.show()