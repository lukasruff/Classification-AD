#!/usr/bin/env bash


mkdir ../log/mnist_oe_emnist;

methods=( hsc bce );
n_classes=( 1 2 3 5 10 15 20 26 );

for seed in $(seq 1 10);
  do
    for k in "${n_classes[@]}";
      do
        for exp in $(seq 0 9);
          do
            for method in "${methods[@]}";
              do
                mkdir ../log/mnist_oe_emnist/${method};
                mkdir ../log/mnist_oe_emnist/${method}/${k}_oe_classes;
                mkdir ../log/mnist_oe_emnist/${method}/${k}_oe_classes/${exp}_vs_rest;
                mkdir ../log/mnist_oe_emnist/${method}/${k}_oe_classes/${exp}_vs_rest/seed_${seed};
                python main.py mnist mnist_LeNet ../log/mnist_oe_emnist/${method}/${k}_oe_classes/${exp}_vs_rest/seed_${seed} ../data --rep_dim 32 --objective ${method} --outlier_exposure True --oe_dataset_name emnist --oe_n_classes ${k} --device cuda --seed ${seed} --lr 0.001 --n_epochs 150 --lr_milestone 50 --lr_milestone 100 --batch_size 128 --normal_class ${exp};
              done
          done
      done
  done
