#!/usr/bin/env bash


mkdir ../log/cifar10_oe_tinyimages;
mkdir ../log/cifar10_oe_tinyimages/focal_sensitivity;

mkdir ../log/imagenet1k_oe_imagenet22k;
mkdir ../log/imagenet1k_oe_imagenet22k/focal_sensitivity;

gammas=( 0 0.5 2 4 );

for seed in $(seq 1 10);
  do
    for exp in $(seq 0 9);
      do
        for gamma in "${gammas[@]}";
          do
            mkdir ../log/cifar10_oe_tinyimages/focal_sensitivity/gamma=${gamma};
            mkdir ../log/cifar10_oe_tinyimages/focal_sensitivity/gamma=${gamma}/${exp}_vs_rest;
            mkdir ../log/cifar10_oe_tinyimages/focal_sensitivity/gamma=${gamma}/${exp}_vs_rest/seed_${seed};
            python main.py cifar10 cifar10_LeNet ../log/cifar10_oe_tinyimages/focal_sensitivity/gamma=${gamma}/${exp}_vs_rest/seed_${seed} ../data --rep_dim 256 --objective focal --focal_gamma ${gamma} --outlier_exposure True --oe_dataset_name tinyimages --device cuda --seed ${seed} --lr 0.001 --n_epochs 200 --lr_milestone 100 --lr_milestone 150 --batch_size 128 --data_augmentation True --data_normalization True --normal_class ${exp};

            mkdir ../log/imagenet1k_oe_imagenet22k/focal_sensitivity/gamma=${gamma};
            mkdir ../log/imagenet1k_oe_imagenet22k/focal_sensitivity/gamma=${gamma}/${exp}_vs_rest;
            mkdir ../log/imagenet1k_oe_imagenet22k/focal_sensitivity/gamma=${gamma}/${exp}_vs_rest/seed_${seed};
            python main.py imagenet1k imagenet_WideResNet ../log/imagenet1k_oe_imagenet22k/focal_sensitivity/gamma=${gamma}/${exp}_vs_rest/seed_${seed} ../data --rep_dim 256 --objective focal --focal_gamma ${gamma} --outlier_exposure True --oe_dataset_name imagenet22k --device cuda --seed ${seed} --lr 0.001 --n_epochs 150 --lr_milestone 100 --lr_milestone 125 --batch_size 128 --data_augmentation True --data_normalization True --normal_class ${exp};
          done
      done
  done
