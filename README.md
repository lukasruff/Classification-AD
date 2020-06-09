# Rethinking Assumptions in Deep Anomaly Detection

This repository provides the code for the methods and experiments presented in our preprint 'Rethinking Assumptions in Deep Anomaly Detection.'

## Citation and Contact

You find a PDF of the paper on arXiv: [https://arxiv.org/abs/2006.00339](https://arxiv.org/abs/2006.00339).
If you find our work useful, please cite:
```
@article{ruff2020rethinking, 
  title   = {Rethinking Assumptions in Deep Anomaly Detection}, 
  author  = {Ruff, Lukas and Vandermeulen, Robert A and Franks, Billy Joe and M{\"u}ller, Klaus-Robert and Kloft, Marius}, 
  journal = {arXiv preprint arXiv:2006.00339}, 
  year    = {2020}
}
```

## Abstract

> > Though anomaly detection (AD) can be viewed as a classification problem (nominal vs. anomalous) it is usually treated in an unsupervised manner since one typically does not have access to, or it is infeasible to utilize, a dataset that sufficiently characterizes what it means to be "anomalous." In this paper we present results demonstrating that this intuition surprisingly does not extend to deep AD on images. For a recent AD benchmark on ImageNet, classifiers trained to discern between normal samples and just a few (64) random natural images are able to outperform the current state of the art in deep AD. We find that this approach is also very effective at other common image AD benchmarks. Experimentally we discover that the multiscale structure of image data makes example anomalies exceptionally
informative.


## Installation
This code is written in `Python 3.7` and requires the packages listed in `requirements.txt`. For running the code, we recommend to set up a virtual environment, e.g. via `virtualenv` or `conda`, and install the packages therein in the specified versions:

### `virtualenv`

```
# pip install virtualenv
cd <path-to-repo>
virtualenv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### `conda`

```
cd <path-to-repo>
conda create --name myenv
conda activate myenv
while read requirement; do conda install -n myenv --yes $requirement; done < requirements.txt
```

## Data

We present experiments using the [MNIST](http://yann.lecun.com/exdb/mnist/), [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset), [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), [80 Million Tiny Images](https://groups.csail.mit.edu/vision/TinyImages/), [ImageNet-1K](http://www.image-net.org/), and [ImageNet-22K](http://www.image-net.org/) datasets in our paper. These datasets get automatically downloaded to the `./data` directory when experiments are run for the first time on the respective datasets, except ImageNet-1K and ImageNet-22K. The ImageNet-1K one-vs-rest anomaly detection benchmark data can be downloaded from [https://github.com/hendrycks/ss-ood](https://github.com/hendrycks/ss-ood), which is the repository of the paper that introduced the benchmark, and should be placed in the `./data/imagenet1k` directory.
The ImageNet-22K dataset can be downloaded from [http://www.image-net.org](http://www.image-net.org/), which requires a registration. Note that our implementation assumes the ImageNet-22K `*.tar` archives to be extracted into the `./data/fall11_whole_extracted` directory.


## Running experiments

All the experiments presented in our paper can be run by using the `main.py` script. The specific method (`hsc`, `deepSAD`, `bce`, or `focal`) can be set via the `--objective` option, e.g. `--objective hsc`.

The `main.py` script features various options and experimental parameters. Have a look into `main.py` for all the possible options and arguments.

Below, we present two examples for the CIFAR-10 as well as the ImageNet one-vs-rest anomaly detection benchmarks. The complete bash scripts to reproduce all experimental results reported in our paper are given in `./src/experiments`.

### CIFAR-10 One-vs-Rest Benchmark using 80 Million Tiny Images as OE

The following runs a Hypersphere Classifier (`--objective hsc`) experiment on CIFAR-10 with class `0` (airplane) considered to be the normal class with using 80 Million Tiny Images as OE (`--oe_dataset_name tinyimages`):

```
cd <path-to-repo>

# activate virtual environment
source myenv/bin/activate  # or 'conda activate myenv' for conda

# create folder for experimental outputs
mkdir log/cifar10_test

# change to source directory
cd src

# run experiment
python main.py cifar10 cifar10_LeNet ../log/cifar10_test ../data --rep_dim 256 --objective hsc --outlier_exposure True --oe_dataset_name tinyimages --device cuda --seed 42 --lr 0.001 --n_epochs 200 --lr_milestone 100 --lr_milestone 150 --batch_size 128 --data_augmentation True --data_normalization True --normal_class 0;
```

### ImageNet-1K One-vs-Rest Benchmark using ImageNet-22K as OE

The following runs a Binary Cross-Entropy Classifier (`--objective bce`) experiment on ImageNet-1K with class `4` (banjo) considered to be the normal class with using ImageNet-22K as OE (`--oe_dataset_name imagenet22k`):

```
cd <path-to-repo>

# activate virtual environment
source myenv/bin/activate  # or 'conda activate myenv' for conda

# create folders for experimental outputs
mkdir log/imagenet_test

# change to source directory
cd src

# run classifier experiment
python main.py imagenet1k imagenet_WideResNet ../log/imagenet_test ../data --rep_dim 256 --objective bce --outlier_exposure True --oe_dataset_name imagenet22k --device cuda --seed 42 --lr 0.001 --n_epochs 150 --lr_milestone 100 --lr_milestone 125 --batch_size 128 --data_augmentation True --data_normalization True --normal_class 4;
```

## License
MIT
