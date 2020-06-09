import json
import torch

from base.base_dataset import BaseADDataset
from networks.main import build_network
from optim import ClassifierTrainer


class Classifier(object):
    """A class for an anomaly detection classifier model.

    Attributes:
        objective: Hypersphere ('hsc'), binary cross-entropy ('bce'), or focal loss ('focal') classifier.
        hsc_norm: Set specific norm to use with HSC ('l1', 'l2', 'l2_squared', 'l2_squared_linear').
        focal_gamma: Gamma parameter of the focal loss.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network.
        trainer: ClassifierTrainer to train the classifier model.
        optimizer_name: A string indicating the optimizer to use for training.
        results: A dictionary to save the results.
    """

    def __init__(self, objective: str, hsc_norm: str = 'l2_squared_linear', focal_gamma: float = 2.0):
        """Inits Classifier."""

        self.objective = objective

        self.hsc_norm = hsc_norm
        self.focal_gamma = focal_gamma

        self.net_name = None
        self.net = None

        self.trainer = None
        self.optimizer_name = None

        self.results = {
            'train_time': None,
            'train_scores': None,
            'test_time': None,
            'test_scores': None,
            'test_auc': None
        }

    def set_network(self, net_name, rep_dim=64, bias_terms=False):
        """Builds the neural network."""
        self.net_name = net_name
        self.net = build_network(net_name, rep_dim=rep_dim, bias_terms=bias_terms)

    def train(self, dataset: BaseADDataset, oe_dataset: BaseADDataset = None, optimizer_name: str = 'adam',
              lr: float = 0.001, n_epochs: int = 50, lr_milestones: tuple = (), batch_size: int = 128,
              weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Trains the classifier on the training data."""

        self.optimizer_name = optimizer_name
        self.trainer = ClassifierTrainer(self.objective, self.hsc_norm, self.focal_gamma, optimizer_name=optimizer_name,
                                         lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones, batch_size=batch_size,
                                         weight_decay=weight_decay, device=device, n_jobs_dataloader=n_jobs_dataloader)

        # Get results
        self.net = self.trainer.train(dataset=dataset, oe_dataset=oe_dataset, net=self.net)
        self.results['train_time'] = self.trainer.train_time
        self.results['train_scores'] = self.trainer.train_scores

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Classifier on the test data."""

        if self.trainer is None:
            self.trainer = ClassifierTrainer(self.objective, self.hsc_norm, self.focal_gamma, device=device,
                                             n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.test(dataset, self.net)

        # Get results
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores
        self.results['test_auc'] = self.trainer.test_auc

    def save_model(self, export_model):
        """Save the classifier model to export_model."""
        net_dict = self.net.state_dict()
        torch.save({'net_dict': net_dict}, export_model)

    def load_model(self, model_path, map_location='cpu'):
        """Load the classifier model from model_path."""
        model_dict = torch.load(model_path, map_location=map_location)
        self.net.load_state_dict(model_dict['net_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
