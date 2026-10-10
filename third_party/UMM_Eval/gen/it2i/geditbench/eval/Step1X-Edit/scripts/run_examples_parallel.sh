GPU=4

# assert GPU = ulysses_degree * ring_degree * cfg_degree
torchrun --nnode=1 --nproc-per-node=$GPU inference.py --input_dir ./examples \
    --model_path /data/work_dir/step1x-edit/ \
    --json_path ./examples/prompt_cn.json \
    --output_dir ./output_cn \
    --seed 1234 --size_level 1024 \
    --ulysses_degree 1 --ring_degree 2 --cfg_degree 2 --teacache

torchrun --nnode=1 --nproc-per-node=$GPU inference.py --input_dir ./examples \
    --model_path /data/work_dir/step1x-edit/ \
    --json_path ./examples/prompt_en.json \
    --output_dir ./output_en \
    --seed 1234 --size_level 1024 \
    --ulysses_degree 1 --ring_degree 2 --cfg_degree 2 --teacache
