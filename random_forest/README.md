# purpose: To build a Random Forest model with an input csv and then save/apply model to a raster
# Inputs: input CSV, Raster Stack

### Train model
python randomforest_andrew.py -dir results -t /att/nobackup/aweis/forMaggie/cloud_training.csv
python rasterRF.py -w results -m newmodel -b 1 2 3 4 5 6 7 8 9 10 11 -bn 'Coastal Blue' Blue Green Yellow Red Red Edge Near-IR1 Near-IR2 DVI FDI SI -c 'results/TrainingData/cloud_training.csv'

### predict cloud
python randomforest_andrew_v2.py -dir results -a results/Models/model_5Images_0002.pkl -i /att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS/8-band/WV02_20140716_M1BS_103001003328DB00-toa.tif

python rasterRF.py -w results -m results/Models/model_20_log2.pkl -b 1 2 3 4 5 6 7 8 9 10 11 -bn 'Coastal Blue' Blue Green Yellow Red Red Edge Near-IR1 Near-IR2 DVI FDI SI


### Test images
/att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/RandomForests/VHR-Stacks/WV02_20140716_M1BS_103001003328DB00-toa_pansharpen.tif - 21G
/att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS/8-band - couple of images
