"""
Purpose: Modify existing csv training data with additional indices and
         sometimes even order.

Data Source: This script has been tested with very high-resolution NGA data.
             Additional testing will be required to validate the applicability
             of this model for other datasets.

Indices Equations:
    - 8 band imagery: FDI := NIR2 - (RedEdge + Blue),
    - 4 band imagery: FDI := NIR1 - (Red + Blue)
    - 8 and 4 band imagery:
    SI := ((factor - Blue) * (factor - Green) * (factor - Red)) ** (1.0 / 3)
    - 8 and 4 band imagery: NDWI := factor * (Green - NIR1) / (Green + NIR1)

Author: Jordan A Caraballo-Vega, Science Data Processing Branch, Code 587
"""
# --------------------------------------------------------------------------------
# Import System Libraries
# --------------------------------------------------------------------------------
import sys
import pandas as pd


if __name__ == "__main__":

    # csv file to read
    csvfile = sys.argv[1]
    print(f"Modifying {csvfile}.")

    # We generate 3 additional CSV training files to train 3 different
    # models. This will help us at the time of classifying data of different
    # structures. The 3 uses cases are listed and called below.

    # initial set of bands included in the original training csv
    bands = [
        'CoastalBlue', 'Blue', 'Green', 'Yellow', 'Red', 'RedEdge',
        'NIR1', 'NIR2', 'DVI', 'FDI', 'SI', 'Class'
    ]
    df = pd.read_csv(csvfile, header=None, sep=',', names=bands)

    # Case #1: cloud_training_8band_fdi_si_ndwi.csv
    # training data using all 8 bands from imagery and the 3 indices
    # calculated using all of the bands.
    fbands = [
        'CoastalBlue', 'Blue', 'Green', 'Yellow', 'Red', 'RedEdge',
        'NIR1', 'NIR2', 'FDI', 'SI', 'NDWI', 'Class'
    ]
    cdrop = ['DVI', 'FDI', 'SI']
    df1 = df.drop(columns=cdrop)

    df1['FDI'] = df1.apply(
        lambda row: row.NIR2 - (row.RedEdge + row.Blue), axis=1
    )
    df1['SI'] = df1.apply(
        lambda row: ((10000 - row.Blue) * (10000 - row.Green)
                     * (10000 - row.Red)) ** (1.0/3),
        axis=1
    )
    df1['NDWI'] = df1.apply(
        lambda row: 10000 * (row.Green - row.NIR1) / (row.Green + row.NIR1),
        axis=1
    )
    df1 = df1.fillna(0)
    df1 = df1.astype({'FDI': 'int16', 'SI': 'int16', 'NDWI': 'int16'})

    df1 = df1[fbands]
    print("shape df: ", df1, df1.shape)
    df1.to_csv(
        'cloud_training_8band_fdi_si_ndwi.csv',
        index=False,
        header=False
    )

    # Case #2: cloud_training_4band_fdi_si_ndwi.csv
    # training data using only 4 bands from imagery and the 3 indices
    # calculated using only 4 bands.
    # The order of the bands goes accordingly to 8 band imagery (B-G-R-NIR).
    fbands = ['Blue', 'Green', 'Red', 'NIR1', 'FDI', 'SI', 'NDWI', 'Class']
    cdrop = ['CoastalBlue', 'Yellow', 'RedEdge', 'NIR2', 'DVI', 'FDI', 'SI']
    df2 = df.drop(columns=cdrop)

    df2['FDI'] = df2.apply(
        lambda row: row.NIR1 - (row.Red + row.Blue), axis=1
    )
    df2['SI'] = df2.apply(
        lambda row: ((10000 - row.Blue) * (10000 - row.Green)
                     * (10000 - row.Red)) ** (1.0/3),
        axis=1
    )
    df2['NDWI'] = df2.apply(
        lambda row: 10000 * (row.Green - row.NIR1) / (row.Green + row.NIR1),
        axis=1
    )
    df2 = df2.fillna(0)
    df2 = df2.astype({'FDI': 'int16', 'SI': 'int16', 'NDWI': 'int16'})

    df2 = df2[fbands]
    print("shape df: ", df2, df2.shape)
    df2.to_csv(
        'cloud_training_4band_fdi_si_ndwi.csv',
        index=False,
        header=False
    )

    # Case #3: cloud_training_4band_rgb_fdi_si_ndwi.csv
    # training data using only 4 bands from imagery and the 3 indices
    # calculated using only 4 bands.
    # The order of the bands was fixed to match (R-G-B-NIR).
    fbands = ['Red', 'Green', 'Blue', 'NIR1', 'FDI', 'SI', 'NDWI', 'Class']
    df3 = df2[fbands]
    print("shape df: ", df3, df3.shape)
    df3.to_csv(
        'cloud_training_4band_rgb_fdi_si_ndwi.csv',
        index=False,
        header=False
    )
