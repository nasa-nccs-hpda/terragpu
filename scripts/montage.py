"""
Montage classified raster scenes.
Author: Jordan A. Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
"""
import glob
import os

# sys.argv[1] - match all tif files to montage
# sys.argv[2] - output file (tiff)

inp = '2dspline_overlap_accuracy/Vietnam_2019228_data_pansharp_*'

file_list = sorted(glob.glob(inp))
print(file_list)
files_string = " ".join(file_list)

command = "python gdal_merge.py -o Vietnam_2019228_data.tif -of gtiff " + \
    files_string
os.system(command)
