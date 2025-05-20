#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : chenht2022
Date         : 2024-07-16 10:43:18
Version      : 1.0.0
LastEditors  : chenht2022 
LastEditTime : 2024-07-25 10:32:53
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import os, sys
import time
import torch
import intel_extension_for_pytorch as ipex
import torch.nn.quantized as nnq

scale, zero_point = 0.1, 0  # Adjust scale and zero_point based on your dataset

hidden_size = 5120
intermediate_size = 3072
layer_num = 10
qlen = 1
warm_up_iter = 1000
test_iter = 10000

def act_fn(x):
    return x / (1.0 + torch.exp(-x))

def mlp_torch(input, gate_proj, up_proj, down_proj):
    if isinstance(gate_proj, nnq.Linear):
        input_q = torch.quantize_per_tensor(input.to(torch.float32), scale, zero_point, torch.quint8)
        gate_buf = gate_proj(input_q)
        up_buf = up_proj(input_q)
        gate_buf = gate_buf.dequantize()
        up_buf = up_buf.dequantize()
        intermediate = act_fn(gate_buf) * up_buf
        intermediate_q = torch.quantize_per_tensor(intermediate, scale, zero_point, torch.quint8)
        expert_output = down_proj(intermediate_q)
        ret = expert_output.dequantize()
    else:
        gate_buf = gate_proj(input)
        up_buf = up_proj(input)
        intermediate = act_fn(gate_buf) * up_buf
        ret = down_proj(intermediate)
    return ret

class simple_mlp(torch.nn.Module):
    def __init__(self, layer_num, hidden_size, intermediate_size, quant_mode, proj_type, device="cpu"):
        super(simple_mlp, self).__init__()
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projs = []
        self.proj_type = proj_type
        for _ in range(layer_num):
            
            if quant_mode == "qint8":
                gate_proj = torch.randn((intermediate_size, hidden_size), dtype=torch.float32, device=device).contiguous()
                up_proj = torch.randn((intermediate_size, hidden_size), dtype=torch.float32, device=device).contiguous()
                down_proj = torch.randn((hidden_size, intermediate_size), dtype=torch.float32, device=device).contiguous()
                gate_proj_q = torch.quantize_per_tensor(gate_proj, scale, zero_point, torch.qint8)
                quantized_gate = nnq.Linear(hidden_size, intermediate_size)
                quantized_gate.set_weight_bias(gate_proj_q, None)
                up_proj_q = torch.quantize_per_tensor(up_proj, scale, zero_point, torch.qint8)
                quantized_up = nnq.Linear(hidden_size, intermediate_size)
                quantized_up.set_weight_bias(up_proj_q, None)
                down_proj_q = torch.quantize_per_tensor(down_proj, scale, zero_point, torch.qint8)
                quantized_down = nnq.Linear(intermediate_size, hidden_size)
                quantized_down.set_weight_bias(down_proj_q, None)
                quantized_gate.to(device)
                quantized_up.to(device)
                quantized_down.to(device)
                self.projs.append((quantized_gate, quantized_up, quantized_down))
            else:
                gate_proj = torch.randn((intermediate_size, hidden_size), dtype=proj_type, device=device).contiguous()
                up_proj = torch.randn((intermediate_size, hidden_size), dtype=proj_type, device=device).contiguous()
                down_proj = torch.randn((hidden_size, intermediate_size), dtype=proj_type, device=device).contiguous()
                gate_linear = torch.nn.Linear(hidden_size, intermediate_size, dtype = proj_type, device=device)
                up_linear = torch.nn.Linear(hidden_size, intermediate_size, dtype = proj_type, device=device)
                down_linear = torch.nn.Linear(intermediate_size, hidden_size, dtype = proj_type, device=device)
                with torch.no_grad():
                    gate_linear.weight.copy_(gate_proj)
                    up_linear.weight.copy_(up_proj)
                    down_linear.weight.copy_(down_proj)
                self.projs.append((gate_linear, up_linear, down_linear))

    def forward(self, input):
        t_output = None
        for i in range(self.layer_num):
            if t_output is None:
                t_output = mlp_torch(input[i], self.projs[i][0], self.projs[i][1], self.projs[i][2])
            else:
                t_output += mlp_torch(input[i], self.projs[i][0], self.projs[i][1], self.projs[i][2])
        return t_output
    
def bench_mlp(quant_mode: str, device: str):
    with torch.inference_mode(mode=True):
        if quant_mode == "fp32":
            proj_type = torch.float32
            bytes_per_elem = 4.000000
        elif quant_mode == "fp16":
            proj_type = torch.float16
            bytes_per_elem = 2.000000
        elif quant_mode == "bf16":
            proj_type = torch.bfloat16
            bytes_per_elem = 2.000000
        elif quant_mode == "qint8":
            proj_type = torch.qint8
            bytes_per_elem = 1.000000
        else:
            assert(False)

        input = torch.randn((layer_num, qlen, hidden_size), dtype=torch.bfloat16, device = device).contiguous()

        model = simple_mlp(layer_num, hidden_size, intermediate_size, quant_mode, proj_type)
        model.eval()
        model = ipex.optimize(model, dtype=torch.bfloat16, weights_prepack=False)
        model = torch.compile(model, backend='ipex')
        model.to(device)
        
        for i in range(warm_up_iter//layer_num):
            with torch.inference_mode():
                output = model(input)
        # test
        start = time.perf_counter()
        for i in range(test_iter//layer_num):
            with torch.inference_mode():
                output = model(input)
            # mlp_torch(input[i % layer_num], gate_projs[i % layer_num], up_projs[i % layer_num], down_projs[i % layer_num])
        end = time.perf_counter()
        total_time = end - start
        print("device: ", device, end=";")
        print('Quant mode: ', quant_mode, end=";")
        print('Time(s): ', total_time, end=";")
        print('Iteration: ', test_iter, end=";") 
        print('Time(us) per iteration: ', total_time / test_iter * 1000000, end=";")
        print('Bandwidth: ', hidden_size * intermediate_size * 3 * bytes_per_elem * test_iter / total_time / 1000 / 1000 / 1000, 'GB/s')
        print('')

#bench_mlp("fp32", "cpu")
bench_mlp("fp16", "cpu")
bench_mlp("bf16", "cpu")
# bench_mlp("qint8", "cpu")
