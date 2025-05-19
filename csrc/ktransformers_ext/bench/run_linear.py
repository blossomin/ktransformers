
python bench_linear.py > ./results/bench_linear_old.log 2>&1
python bench_linear_torch.py > ./results/bench_linear_torch_old.log 2>&1
python bench_linear_torch_new.py > ./results/bench_linear_torch_new.log 2>&1
python bench_linear_amx.py > ./results/bench_attention_amx_new_bf16.log 2>&1