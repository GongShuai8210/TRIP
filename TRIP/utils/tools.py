from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table, flop_count_str
from numbers import Number
from typing import Any, Callable, List, Optional, Union
import numpy as np
import torch

def print_color_text(text, color):
    color_dice = {'black': '30', 'red': '31', 'green': '32', 'orange': '33',
                  'blue': '34', 'purple': '35', 'blue-green': '36', 'white': '37'}

    print("\033[{}m{}\033[0m".format(color_dice[color], text))

def rfft_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the rfft/rfftn operator.
    """
    input_shape = inputs[0].type().sizes()
    B, H, W, C = input_shape
    N = H * W
    flops = N * C * np.ceil(np.log2(N))
    return flops

def count_flops(model, ratios=None):
    with torch.no_grad():
        # x_s = torch.randn(1, 3, 224, 224)
        # x_t = torch.randn(1, 3, 224, 224)
        # x = torch.cat((x_s, x_t), dim=0)
        x = torch.randn(1, 3, 224, 224)
        input_var = x.cuda()
        model.default_ratio = ratios
        flops = FlopCountAnalysis(model, input_var)
        handlers = {
            'aten::fft_rfft2': rfft_flop_jit,
            'aten::fft_irfft2': rfft_flop_jit,
        }
        flops.set_op_handle(**handlers)
        flops_all = flops.total()
        print(flop_count_table(flops, max_depth=4))
        print(flop_count_str(flops))
        print("#### GFLOPs: {} for ratio {}".format(flops_all / 1e9, ratios))
    return flops_all / 1e9

