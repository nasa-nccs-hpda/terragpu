# Script to fix training data to work with 4 band model
# modifies existing csv file provided by Andrew and company
import pandas as pd

# csv file to read
csvfile = 'cloud_training.csv'

# initial and final set of bands
bands = ['CoastalBlue', 'Blue', 'Green', 'Yellow', 'Red', 'RedEdge', 'NearIR1', 'NearIR2', 'DVI', 'FDI', 'SI', 'Class']
fbands = ['Blue', 'Green', 'Red', 'NearIR1', 'DVI', 'FDI', 'SI', 'CI1', 'CI2', 'Class']

# generate pandas data frame
df = pd.read_csv(csvfile, header=None, sep=',', names=bands)  # generate pd dataframe

# drop undesired columns
df = df.drop(columns=['CoastalBlue', 'Yellow', 'RedEdge', 'NearIR2', 'DVI', 'FDI', 'SI'])

# calculate indices - DVI
df['DVI'] = df.apply(lambda row: row.NearIR1 - row.Red, axis=1)

# calculate indices - FDI
df['FDI'] = df.apply(lambda row: row.NearIR1 - (row.Red + row.Blue), axis=1)

# calculate indices - SI
df['SI'] = df.apply(lambda row: (1 - row.Blue) * (1 - row.Green) * (1 - row.Red), axis=1)

# calculate indices - two indices from paper
# CI1 = (3. * NIR) / (B + G + R), CI2 = (B + G + R + NIR) / 4.
df['CI1'] = df.apply(lambda row: (3. * row.NearIR1) / (row.Blue + row.Green + row.Red), axis=1)
df['CI2'] = df.apply(lambda row: (row.Blue + row.Green + row.Red + row.NearIR1) / 4., axis=1)

df = df[fbands]

print("shape df: ", df, df.shape, type(df))

df.to_csv('cloud_training_4band_ci.csv', index=False)