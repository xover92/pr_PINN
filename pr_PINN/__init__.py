#!/usr/bin/python
# -*- coding: utf-8 -*-

from .__version__ import __version__
from pr_PINN.pinn import generate_plot, training_loop, solve_with_fipy

__author__ = ['Francesco Colombo']
__email__ = ['francesco.colombo10@studio.unibo.it']

__all__ = [
    '__version__',
    'generate_plot',
    'training_loop',
    'solve_with_fipy'
]
