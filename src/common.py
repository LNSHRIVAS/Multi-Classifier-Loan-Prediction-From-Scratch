import numpy as np
import pandas as pd
import unittest
import logging
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import logging
import sys
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.info("info")