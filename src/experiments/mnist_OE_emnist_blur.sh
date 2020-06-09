#!/usr/bin/env bash


mkdir ../log/mnist_oe_emnist;
mkdir ../log/mnist_oe_emnist/blur;

methods=( hsc bce );
stds=( 1 2 4 8 16 32 );

for seed in $(seq 1 5);
  do
    for exp in $(seq 0 9);
      do
        for std in "${stds[@]}";
          do
            for method in "${methods[@]}";
              do
                mkdir ../log/mnist_oe_emnist/blur/${method};
                mkdir ../log/mnist_oe_emnist/blur/${method}/blur_std=${std};
                mkdir ../log/mnist_oe_emnist/blur/${method}/blur_std=${std}/${exp}_vs_rest;
                mkdir ../log/mnist_oe_emnist/blur/${method}/blur_std=${std}/${exp}_vs_rest/seed_${seed};
                python main.py mnist mnist_LeNet ../log/mnist_oe_emnist/blur/${method}/blur_std=${std}/${exp}_vs_rest/seed_${seed} ../data --rep_dim 32 --objective ${method} --outlier_exposure True --oe_dataset_name emnist --oe_n_classes 26 --blur_oe True --blur_std ${std} --device cuda --seed ${seed} --lr 0.001 --n_epochs 150 --lr_milestone 50 --lr_milestone 100 --batch_size 128 --normal_class ${exp};
              done
          done
      done
  done
