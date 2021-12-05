# -*- coding: utf-8 -*-
# RF pipeline: preprocess, train, and predict.

import sys
import argparse
import logging

from omegaconf import OmegaConf
from terragpu import rf_model

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"


# -----------------------------------------------------------------------------
# main
#
# python rf_pipeline.py options here
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Use this application to perform classifications with Random Forest'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--cfg", type=str, required=True, dest='config_file',
        help="YML filename to store training")

    parser.add_argument(
        '--step', type=str, nargs='*', required=True,
        dest='pipeline_step', help='Pipeline step to perform',
        default=['preprocess', 'train', 'predict', 'vis'],
        choices=['preprocess', 'train', 'predict', 'vis'])

    args = parser.parse_args()

    # --------------------------------------------------------------------------------
    # Set logging
    # --------------------------------------------------------------------------------
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)  # set stdout handler
    ch.setLevel(logging.INFO)

    # Set formatter and handlers
    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # --------------------------------------------------------------------------------
    # Initialiaze pipeline
    # --------------------------------------------------------------------------------
    conf = OmegaConf.load(args.config_file)
    pipeline = rf_model.RF(
        train_csv=conf.train_csv, dataset_metadata=conf.dataset_metadata,
        model_metadata=conf.model_metadata, model_filename=conf.model_filename,
        output_dir=conf.output_dir, predict_dir=conf.predict_dir
    )

    # --------------------------------------------------------------------------------
    # Execute pipeline step
    # --------------------------------------------------------------------------------
    if "preprocess" in args.pipeline_step:
        pipeline.preprocess()
    #elif args.pipeline_step == "train":
    #    pipeline.train()
    #elif args.pipeline_step == "predict":
    #    pipeline.predict()

    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
