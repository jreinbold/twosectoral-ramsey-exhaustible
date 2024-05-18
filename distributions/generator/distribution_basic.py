import csv
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap



# =============================================================================
# define parameters
# =============================================================================

num_entries = 1000
mean = 100
sigma_1 = 30
sigma_2 = 15
min_value = 10
max_value = 190

# =============================================================================
# generate random values for productivity and initial wealth
# =============================================================================

initial_wealth = [(max(min(random.gauss(mean, sigma_1), max_value), min_value))/100 for _ in range(num_entries)]
productivity = [(max(min(random.gauss(mean, sigma_2), max_value), min_value))/100 for _ in range(num_entries)]


data = {'productivity': productivity,'initial_wealth': initial_wealth}
df = pd.DataFrame(data)

# Visualize existing distribtuion
df = pd.read_csv("path")

# Save newly generated distirbution
#df.to_csv('basic_distribution.csv', index=False)


# =============================================================================
# Single plot distributions
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].hist(df['initial_wealth'], bins=20, alpha=0.7)
axes[0].set_xlabel('Initial wealth')
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

print(df)


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
ax.set_xlabel('Productivity')
ax.set_ylabel('Initial wealth')
ax.set_zlabel('Frequency in %')

plt.show()

# =============================================================================
# Combined plots
# =============================================================================

fig = plt.figure(figsize=plt.figaspect(0.33))

# First subplot
ax = fig.add_subplot(1, 3, 1)

ax.hist(df['productivity'], bins=50, alpha=0.7)
ax.set_xlabel('Productivity')
ax.set_ylabel('Frequency')
ax.set_title("(a) Productivity distribution", y=-0.3)

# Second subplot
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
# plt.savefig('distribution_basic.png', bbox_inches='tight',pad_inches = 0.3,dpi = 200)
plt.show()