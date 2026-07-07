#!/usr/bin/python
# -*- coding: utf-8 -*-

import pr_PINN.pr_pinn as prp
import torch
import gradio as gr
import pr_PINN.pr_prinn2d as prp2
import pr_PINN.pr_pinn3d as prp3

__author__ = ['Francesco Colombo']
__email__ = ['francesco.colombo10@studio.unibo.it']

torch.manual_seed(42)


def choose_dimensions(n_epochs, n_neurons, n_dimensions, n_points):
    if n_dimensions == 1:
        fig = prp.generate_plot_1d(n_epochs, n_neurons, n_points)
    if n_dimensions == 2:
        fig = prp2.generate_plot_2d(n_epochs, n_neurons, n_points)
    if n_dimensions == 3:
        fig = prp3.generate_plot_3d(n_epochs, n_neurons, n_points)
    return fig


demo = gr.Interface(
    fn=choose_dimensions,
    inputs=[gr.Slider(maximum=1000), gr.Slider(),
            gr.Slider(minimum=1, maximum=3, step=1.0),
            gr.Slider(maximum=5000)],
    outputs=gr.Plot(),
    api_name="predict"
)

demo.launch()
