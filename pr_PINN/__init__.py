#!/usr/bin/python
# -*- coding: utf-8 -*-

from .__version__ import __version__
from pr_PINN.pr_pinn import PINN_1d
from pr_PINN.pr_prinn2d import PINN_2d

__author__ = ['Francesco Colombo']
__email__ = ['francesco.colombo10@studio.unibo.it']

__all__ = [
    '__version__',
    'PINN_1d',
    'PINN_2d'
]
