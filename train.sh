#!/bin/bash

source /home/gongshuai/anaconda3/bin/activate pFL

export PYTHONPATH=/home/gongshuai/code/FedDG/fedclip:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
#source /path/to/conda/bin/activate YOUR_CONDA_ENVIRONMENT

for train_ca in 1.0
do
for infer_ca in  2.0
do
for gamma in 0.8
do
for lr in  5e-4
do      

	for domain in 0 1  2 3
        do	
	for num_experts in  8
        do
	for N_CTX in  32
	do
        echo "Running with domain = $domain,NCX $N_CTX num_experts = $num_experts, lr=$lr train_ca=$train_ca,gamma=$gamma,infer_ca=$infer_ca"

         python ../methods/FedMoeP_NR_cos.py --dataset pacs  --mode FedAtImg --test_envs $domain --iters 15   --wk_iters 1   --root_dir '/home/gongshuai/code/FedDG/data/' --batch 16  --num_experts $num_experts --N_CTX $N_CTX  --lr $lr --optimizers AdamW  --weight_decay 0.01  --eps 1e-7  --route_emb 768  --WARMUP_EPOCH 2 --WARMUP_CONS_LR 1e-5 --capacity_factor_train $train_ca --weight_dir /home/gongshuai/code/FedDG/fedclip/weight_dir/ --seed 42 --gamma $gamma --capacity_factor_infer $infer_ca
done	 
done
done
done
done
done
done
