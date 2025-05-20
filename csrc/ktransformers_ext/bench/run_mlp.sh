
python bench_mlp.py > ./results/bench_mlp_old.log 2>&1
python bench_mlp_torch.py > ./results/bench_mlp_torch_old.log 2>&1
python bench_mlp_torch_new.py > ./results/bench_mlp_torch_new.log 2>&1
python bench_mlp_amx.py > ./results/bench_mlp_amx_new_bf16.log 2>&1