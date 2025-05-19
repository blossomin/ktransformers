
python bench_attention.py > ./results/bench_attention_old.log 2>&1
python bench_attention_torch.py > ./results/bench_attention_torch_old.log 2>&1
python bench_attention_torch_new.py > ./results/bench_attention_torch_new.log 2>&1
python bench_attention_amx.py > ./results/bench_attention_amx_new_bf16.log 2>&1