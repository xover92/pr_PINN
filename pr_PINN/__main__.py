#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import gradio as gr
import pr_PINN.pinn as pinn

__author__ = ['Francesco Colombo']
__email__ = ['francesco.colombo10@studio.unibo.it']

torch.manual_seed(42)


# def choose_dimensions(n_epochs, n_neurons, n_dimensions,
#                  n_points, value_x0, value_x1, value_t0):
# fig = pinn.generate_plot(
#    n_epochs, n_neurons, n_points, n_dimensions, value_x0,
#    value_x1, value_t0)
# return fig


demo = gr.Interface(
    fn=pinn.generate_plot,
    inputs=[gr.Slider(maximum=10000), gr.Slider(),
            gr.Slider(maximum=100000),
            gr.Slider(minimum=1, maximum=3, step=1.0),
            gr.Slider(minimum=0.0, maximum=1.0, step=0.01),
            gr.Slider(minimum=0.0, maximum=1.0, step=0.01),
            gr.Slider(minimum=0.0, maximum=1.0, step=0.01),
            gr.Slider(minimum=0.0, maximum=1.0, step=0.01),
            gr.Slider(minimum=0.0, maximum=1.0, step=0.01),
            gr.Slider(minimum=0.0, maximum=1.0, step=0.01),
            gr.Slider(minimum=0.0, maximum=1.0, step=0.01)],
    outputs=[gr.Plot(label='Solution comparison and loss progression'),
             gr.Text(label='L2 loss')],
    api_name="predict"
)

demo.launch()
