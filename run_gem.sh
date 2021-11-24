#!/bin/bash

MY_PYTHON="python"
MNIST_ROTA="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 1000 --data_file mnist_rotations.pt    --cuda no  --seed 0"
MNIST_PERM="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 1000 --data_file mnist_permutations.pt --cuda no  --seed 0"
CIFAR_100i="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 2500 --data_file cifar100.pt           --cuda yes --seed 0"

# model "GEM"
$MY_PYTHON main.py $MNIST_ROTA --model gem --lr 0.01 --n_memories 256 --memory_strength 0.5
#$MY_PYTHON main.py $MNIST_PERM --model gem --lr 0.1 --n_memories 256 --memory_strength 0.5
#$MY_PYTHON main.py $CIFAR_100i --model gem --lr 0.1 --n_memories 256 --memory_strength 0.5

# plot results
cd results/
$MY_PYTHON plot_results.py
cd ..
