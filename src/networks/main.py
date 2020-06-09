from .mnist_LeNet import MNIST_LeNet
from .cifar10_LeNet import CIFAR10_LeNet
from .imagenet_WideResNet import ImageNet_WideResNet
from .toy_Net import Toy_Net
from .classifier_Net import ClassifierNet


def build_network(net_name, rep_dim=64, bias_terms=False):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'imagenet_WideResNet', 'toy_Net',
                            'mnist_LeNet_classifier', 'cifar10_LeNet_classifier', 'imagenet_WideResNet_classifier',
                            'toy_Net_classifier')
    assert net_name in implemented_networks

    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet(rep_dim=rep_dim, bias_terms=bias_terms)

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet(rep_dim=rep_dim, bias_terms=bias_terms)

    if net_name == 'imagenet_WideResNet':
        net = ImageNet_WideResNet(rep_dim=rep_dim)

    if net_name == 'toy_Net':
        net = Toy_Net(rep_dim=rep_dim)

    if net_name in ['mnist_LeNet_classifier', 'cifar10_LeNet_classifier', 'imagenet_WideResNet_classifier',
                    'toy_Net_classifier']:
        net = ClassifierNet(net_name, rep_dim=rep_dim, bias_terms=bias_terms)

    return net
