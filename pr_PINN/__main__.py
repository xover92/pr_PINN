#!/usr/bin/python
# -*- coding: utf-8 -*-

import pr_PINN.pr_pinn as prp
import torch
import gradio as gr

__author__ = ['Francesco Colombo']
__email__ = ['francesco.colombo10@studio.unibo.it']

torch.manual_seed(42)

prp.model = None
prp.optimizer = None


demo = gr.Interface(
    fn=prp.generate_plot_1d,
    inputs=[gr.Slider(maximum=1000), gr.Slider()],
    outputs=gr.Plot(),
    api_name="predict"
)

demo.launch()
