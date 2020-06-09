#!/usr/bin/env bash


mkdir ../log/imagenet1k_oe_imagenet22k;

methods=( hsc deepSAD bce focal);

for seed in $(seq 1 10);
  do
    for exp in $(seq 0 29);
      do
        for method in "${methods[@]}";
          do
            mkdir ../log/imagenet1k_oe_imagenet22k/${method};
            mkdir ../log/imagenet1k_oe_imagenet22k/${method}/${exp}_vs_rest;
            mkdir ../log/imagenet1k_oe_imagenet22k/${method}/${exp}_vs_rest/seed_${seed};
            python main.py imagenet1k imagenet_WideResNet ../log/imagenet1k_oe_imagenet22k/${method}/${exp}_vs_rest/seed_${seed} ../data --rep_dim 256 --objective ${method} --outlier_exposure True --oe_dataset_name imagenet22k --device cuda --seed ${seed} --lr 0.001 --n_epochs 150 --lr_milestone 100 --lr_milestone 125 --batch_size 128 --data_augmentation True --data_normalization True --normal_class ${exp};
          done
      done
  done
