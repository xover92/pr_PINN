#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import gradio as gr
import pr_PINN.pinn as pinn

__author__ = ['Francesco Colombo']
__email__ = ['francesco.colombo10@studio.unibo.it']

torch.manual_seed(42)


def choose_dimensions(n_epochs, n_neurons, n_dimensions, n_points):
    fig = pinn.generate_plot(n_epochs, n_neurons, n_points, n_dimensions)
    return fig


demo = gr.Interface(
    fn=choose_dimensions,
    inputs=[gr.Slider(maximum=10000), gr.Slider(),
            gr.Slider(minimum=1, maximum=3, step=1.0),
            gr.Slider(maximum=10000)],
    outputs=[gr.Plot(label='Solution comparison and loss progression'),
             gr.Text(label='L2 loss')],
    api_name="predict"
)

demo.launch()
