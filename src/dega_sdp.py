"""Compatibility exports for DEGA-SDP.

The project is split into config, data, model, and experiment utility modules.
This file re-exports the public API used by run_experiments.py.
"""

from .config import *
from .data import *
from .experiment_utils import *
from .model import *
