# purpose: To build a Random Forest model with an input csv and then save/apply model to a raster
# Inputs: input CSV, Raster Stack

### Train model
python randomforest_andrew.py -dir results -t /att/nobackup/aweis/forMaggie/cloud_training.csv

### predict cloud
python randomforest_andrew_v2.py -dir results -a results/Models/model_5Images_0002.pkl -i /att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS/8-band/WV02_20140716_M1BS_103001003328DB00-toa.tif
