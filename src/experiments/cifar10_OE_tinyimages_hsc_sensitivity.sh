#!/usr/bin/env bash


mkdir ../log/cifar10_oe_tinyimages;
mkdir ../log/cifar10_oe_tinyimages/hsc_sensitivity;
mkdir ../log/cifar10_oe_tinyimages/hsc_sensitivity/data_augment;
mkdir ../log/cifar10_oe_tinyimages/hsc_sensitivity/no_data_augment;

norms=( l1 l2 l2_squared l2_squared_linear );

for seed in $(seq 1 10);
  do
    for exp in $(seq 0 9);
      do
        for norm in "${norms[@]}";
          do
            mkdir ../log/cifar10_oe_tinyimages/hsc_sensitivity/data_augment;
            mkdir ../log/cifar10_oe_tinyimages/hsc_sensitivity/data_augment/${norm};
            mkdir ../log/cifar10_oe_tinyimages/hsc_sensitivity/data_augment/${norm}/${exp}_vs_rest;
            mkdir ../log/cifar10_oe_tinyimages/hsc_sensitivity/data_augment/${norm}/${exp}_vs_rest/seed_${seed};
            python main.py cifar10 cifar10_LeNet ../log/cifar10_oe_tinyimages/hsc_sensitivity/data_augment/${norm}/${exp}_vs_rest/seed_${seed} ../data --rep_dim 256 --objective hsc --hsc_norm ${norm} --outlier_exposure True --oe_dataset_name tinyimages --device cuda --seed ${seed} --lr 0.001 --n_epochs 200 --lr_milestone 100 --lr_milestone 150 --batch_size 128 --data_augmentation True --data_normalization True --normal_class ${exp};

            mkdir ../log/cifar10_oe_tinyimages/hsc_sensitivity/no_data_augment/${norm};
            mkdir ../log/cifar10_oe_tinyimages/hsc_sensitivity/no_data_augment/${norm}/${exp}_vs_rest;
            mkdir ../log/cifar10_oe_tinyimages/hsc_sensitivity/no_data_augment/${norm}/${exp}_vs_rest/seed_${seed};
            python main.py cifar10 cifar10_LeNet ../log/cifar10_oe_tinyimages/hsc_sensitivity/no_data_augment/${norm}/${exp}_vs_rest/seed_${seed} ../data --rep_dim 256 --objective hsc --hsc_norm ${norm} --outlier_exposure True --oe_dataset_name tinyimages --device cuda --seed ${seed} --lr 0.001 --n_epochs 200 --lr_milestone 100 --lr_milestone 150 --batch_size 128 --data_augmentation False --data_normalization True --normal_class ${exp};
          done
      done
  done
