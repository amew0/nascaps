# CUDA_VISIBLE_DEVICES=1 python main.py --epochs 5 --output out_mnist-2023-06-17-16-19 --population 10 --offsprings 10 --generations 50 --timeout 300 --save_dir results/data --cache_dir results/cache 2>results/logs/mnist-2023-06-17-16-19.err >results/logs/mnist-2023-06-17-16-19.std
dat="mnist-$(date +"%F-%H-%M")"
touch results/logs/${dat}.err
touch results/logs/${dat}.std
CUDA_VISIBLE_DEVICES=1 python main.py  --output out_${dat} --save_dir results/data --cache_dir results/cache 2>results/logs/${dat}.err >results/logs/${dat}.std
