#!/usr/bin/env bash


mkdir ../log/imagenet1k_oe_imagenet22k;
mkdir ../log/imagenet1k_oe_imagenet22k/blur;

methods=( hsc bce );
stds=( 1 2 4 8 16 32 );

for seed in $(seq 1 5);
  do
    for std in "${stds[@]}";
      do
        for exp in $(seq 0 29);
          do
            for method in "${methods[@]}";
              do
                mkdir ../log/imagenet1k_oe_imagenet22k/blur/${method};
                mkdir ../log/imagenet1k_oe_imagenet22k/blur/${method}/blur_std=${std};
                mkdir ../log/imagenet1k_oe_imagenet22k/blur/${method}/blur_std=${std}/${exp}_vs_rest;
                mkdir ../log/imagenet1k_oe_imagenet22k/blur/${method}/blur_std=${std}/${exp}_vs_rest/seed_${seed};
                python main.py imagenet1k imagenet_WideResNet ../log/imagenet1k_oe_imagenet22k/blur/${method}/blur_std=${std}/${exp}_vs_rest/seed_${seed} ../data --rep_dim 256 --objective ${method} --outlier_exposure True --oe_dataset_name imagenet22k --oe_size 64 --blur_oe True --blur_std ${std} --device cuda --seed ${seed} --lr 0.001 --n_epochs 150 --lr_milestone 100 --lr_milestone 125 --batch_size 128 --data_augmentation True --data_normalization True --normal_class ${exp};
              done
          done
      done
  done
