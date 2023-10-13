import os
import warnings

import numpy as np
import torch
from torch.nn.modules.utils import _pair, _single, _triple
from torch.onnx.symbolic_helper import (
    _get_tensor_dim_size,
    _get_tensor_sizes,
    parse_args,
)

@parse_args('v', 'v', 'i', 'i', 'i')
def grid_sampler(g,
                 input,
                 grid,
                 interpolation_mode,
                 padding_mode,
                 align_corners=False):
    output = g.op(
        'mmcv::grid_sampler',
        input,
        grid,
        interpolation_mode_i=interpolation_mode,
        padding_mode_i=padding_mode,
        align_corners_i=align_corners,
    )
    input_shape = _get_tensor_sizes(input)
    if input_shape and hasattr(grid.type(), 'with_sizes'):
        output_type = grid.type().with_sizes(
            [
                _get_tensor_dim_size(input, 0),
                _get_tensor_dim_size(input, 1),
                _get_tensor_dim_size(grid, 1),
                _get_tensor_dim_size(grid, 2),
            ]
        )
        output.setType(output_type)
    return output

def register_extra_symbolics(opset=11):
    # Following strings of text style are from colorama package
    torch.onnx.register_custom_op_symbolic('::grid_sampler', grid_sampler, opset)
