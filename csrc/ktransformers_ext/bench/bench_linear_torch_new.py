#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : chenht2022
Date         : 2024-07-25 10:31:59
Version      : 1.0.0
LastEditors  : chenht2022 
LastEditTime : 2024-07-25 10:32:48
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import os, sys
import time
import torch
import torch.nn.quantized as nnq
import intel_extension_for_pytorch as ipex

scale, zero_point = 0.1, 0  # Adjust scale and zero_point based on your dataset

input_size = 16384
output_size = 5120
layer_num = 10
qlen = 1
warm_up_iter = 1000
test_iter = 10000

class simple_linear(torch.nn.Module):
    def __init__(self, layer_num, input_size, output_size, quant_mode, proj_type, device):
        super(simple_linear, self).__init__()
        self.layer_num = layer_num
        self.input_size = input_size
        self.output_size = output_size
        self.projs = []
        self.quant_mode = quant_mode
        self.proj_type = proj_type
        self.device = device
        for _ in range(layer_num):
            if quant_mode == "qint8":
                proj = torch.randn((output_size, input_size), dtype = torch.float32, device = device).contiguous()
                proj_q = torch.quantize_per_tensor(proj, scale, zero_point, torch.qint8)
                quantized_layer = nnq.Linear(input_size, output_size)
                quantized_layer.set_weight_bias(proj_q, None)
                quantized_layer.to(device)
                self.projs.append(quantized_layer)
            else:
                proj = torch.randn((output_size, input_size), dtype = self.proj_type, device = device).contiguous()
                layer = torch.nn.Linear(input_size, output_size, dtype = self.proj_type, device = device)
                with torch.no_grad():
                    layer.weight.copy_(proj)
                self.projs.append(layer)

    def forward(self, input):
        t_output = None
        for i in range(self.layer_num):
            if self.quant_mode == "qint8":
                input_q = torch.quantize_per_tensor(input[i].to(torch.float32), scale, zero_point, torch.quint8)
                if t_output is None:
                    t_output = self.projs[i](input_q)
                else:
                    t_output += self.projs[i](input_q)
            else:
                # print(f"device of proj: {self.projs[i].weight.device}, device of input: {input[i].device}")
                if t_output is None:
                    t_output = self.projs[i](input[i])
                else:
                    t_output += self.projs[i](input[i])
        return t_output

def bench_linear(quant_mode: str, device: str):
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

        input = torch.randn((layer_num, qlen, input_size), dtype=proj_type, device = "cuda").to(device).contiguous()

        model = simple_linear(layer_num, input_size, output_size, quant_mode, proj_type, device)
        model = model.eval().to(device)
        # model = ipex.optimize(model, dtype=torch.bfloat16)
        
        # warm up
        for i in range(warm_up_iter//layer_num):
            with torch.inference_mode():
                output = model(input)
        # test
        start = time.perf_counter()
        for i in range(test_iter//layer_num):
            with torch.inference_mode():
                output = model(input)
        end = time.perf_counter()
        total_time = end - start
        print("device: ", device, end=";")
        print('Quant mode: ', quant_mode, end=";")
        print('Time(s): ', total_time, end=";")
        print('Iteration: ', test_iter, end=";") 
        print('Time(us) per iteration: ', total_time / test_iter * 1000000, end=";")
        print('Bandwidth: ', input_size * output_size * bytes_per_elem * test_iter / total_time / 1000 / 1000 / 1000, 'GB/s')
        print('')

bench_linear("fp32", "cpu")
bench_linear("fp16", "cpu")
bench_linear("bf16", "cpu")
# bench_linear("qint8", "cpu")

bench_linear("fp32", "cuda")
bench_linear("fp16", "cuda")
bench_linear("bf16", "cuda")
# bench_linear("qint8", "cuda")
