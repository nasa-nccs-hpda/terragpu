# -*- coding: utf-8 -*-
# RF pipeline: preprocess, train, and predict.

import sys
import argparse
import logging

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"


# -----------------------------------------------------------------------------
# main
#
# python rf_pipeline.py options here
# -----------------------------------------------------------------------------
class RFWidgets(object):

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(
            self,
            notebook: bool = True,
        ):

        # Set data directory values
        self.notebook = notebook


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    # placeholder for RFWidgets class
    logging.info("Upcoming Jupyter Notebook visualizations")
