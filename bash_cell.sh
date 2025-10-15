#!/bin/bash
nohup python code/classification/train.py -c classification/config/EL-ResNet18.txt --output_dir classification/output_CUB_hybrid --enable_nmf  --world_size 1 --num_runs 1 > output_CUB_hybrid.log 2>&1 &

nohup python code/classification/train.py -c classification/config/L-ResNet18.txt --output_dir classification/output_CUB_lorentz --enable_nmf  --world_size 1  --num_runs 1 > output_CUB_lorentz.log 2>&1 &

nohup python code/classification/train.py -c classification/config/E-ResNet18.txt --output_dir classification/output_CUB_euclid --enable_nmf  --world_size 1 --num_runs 1 > output_CUB_euclid.log 2>&1 &

nohup python code/classification/train.py -c classification/config/error-ResNet18.txt --output_dir classification/output_CUB_error --enable_nmf  --world_size 1 --num_runs 1 > output_CUB_error.log 2>&1 &





nohup python code/classification/train.py -c classification/config/EL-ResNet18.txt --output_dir classification/output_CUB_hybrid_without_nmf   --world_size 1 --num_runs 1 > output_CUB_hybrid_without_nmf.log 2>&1 &

nohup python code/classification/train.py -c classification/config/L-ResNet18.txt --output_dir classification/output_CUB_lorentz_without_nmf   --world_size 2  --num_runs 1 > output_CUB_lorentz_without_nmf.log 2>&1 &

nohup python code/classification/train.py -c classification/config/E-ResNet18.txt --output_dir classification/output_CUB_euclid_without_nmf   --world_size 1 --num_runs 1 > output_CUB_euclid_without_nmf.log 2>&1 &

nohup python code/classification/train.py -c classification/config/error-ResNet18.txt --output_dir classification/output_cifar10_error_without_nmf --world_size 1 --num_runs 1 > output_cifar10_error_without_nmf.log 2>&1 &
