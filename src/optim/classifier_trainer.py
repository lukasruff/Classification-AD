from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from networks.modules.focal_loss import FocalLoss
from torch.utils.data import DataLoader, RandomSampler
from sklearn.metrics import roc_auc_score

import logging
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ClassifierTrainer(BaseTrainer):

    def __init__(self, objective: str, hsc_norm: str = 'l2_squared_linear', focal_gamma: float = 2.0,
                 optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Classifier parameters
        self.objective = objective
        self.hsc_norm = hsc_norm
        self.focal_gamma = focal_gamma
        self.eps = 1e-9

        # Results
        self.train_time = None
        self.train_scores = None
        self.test_time = None
        self.test_scores = None
        self.test_auc = None

    def train(self, dataset: BaseADDataset, oe_dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Get train data loader
        if oe_dataset is not None:
            num_workers = int(self.n_jobs_dataloader / 2)
        else:
            num_workers = self.n_jobs_dataloader

        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=num_workers)
        if oe_dataset is not None:
            if oe_dataset.shuffle:
                if len(dataset.train_set) > len(oe_dataset.train_set):
                    oe_sampler = RandomSampler(oe_dataset.train_set,
                                               replacement=True, num_samples=len(dataset.train_set))
                    oe_loader = DataLoader(dataset=oe_dataset.train_set, batch_size=self.batch_size, shuffle=False,
                                           sampler=oe_sampler, num_workers=num_workers, drop_last=True)
                else:
                    oe_loader = DataLoader(dataset=oe_dataset.train_set, batch_size=self.batch_size, shuffle=True,
                                           num_workers=num_workers, drop_last=True)

            else:
                oe_loader = DataLoader(dataset=oe_dataset.train_set, batch_size=self.batch_size, shuffle=False,
                                       num_workers=num_workers, drop_last=True)
            dataset_loader = zip(train_loader, oe_loader)
        else:
            dataset_loader = train_loader

        # Set loss
        if self.objective in ['bce', 'focal']:
            if self.objective == 'bce':
                criterion = nn.BCEWithLogitsLoss()
            if self.objective == 'focal':
                criterion = FocalLoss(gamma=self.focal_gamma)
            criterion = criterion.to(self.device)

        # Set device
        net = net.to(self.device)

        # Set optimizer
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting training...')
        net.train()
        start_time = time.time()

        for epoch in range(self.n_epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            idx_label_score = []
            epoch_start_time = time.time()

            # start at random point for the outlier exposure dataset in each epoch
            if (oe_dataset is not None) and (epoch < self.n_epochs):
                oe_loader.dataset.offset = np.random.randint(len(oe_loader.dataset))
                if oe_loader.dataset.shuffle_idxs:
                    random.shuffle(oe_loader.dataset.idxs)
                dataset_loader = zip(train_loader, oe_loader)

            # only load samples from the original training set in a last epoch for saving train scores
            if epoch == self.n_epochs:
                dataset_loader = train_loader
                net.eval()

            for data in dataset_loader:
                if (oe_dataset is not None) and (epoch < self.n_epochs):
                    inputs = torch.cat((data[0][0], data[1][0]), 0)
                    labels = torch.cat((data[0][1], data[1][1]), 0)
                    semi_targets = torch.cat((data[0][2], data[1][2]), 0)
                    idx = torch.cat((data[0][3], data[1][3]), 0)
                else:
                    inputs, labels, semi_targets, idx = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)
                idx = idx.to(self.device)

                # Zero the network parameter gradients
                if epoch < self.n_epochs:
                    optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)

                if self.objective == 'hsc':
                    if self.hsc_norm == 'l1':
                        dists = torch.norm(outputs, p=1, dim=1)
                    if self.hsc_norm == 'l2':
                        dists = torch.norm(outputs, p=2, dim=1)
                    if self.hsc_norm == 'l2_squared':
                        dists = torch.norm(outputs, p=2, dim=1) ** 2
                    if self.hsc_norm == 'l2_squared_linear':
                        dists = torch.sqrt(torch.norm(outputs, p=2, dim=1) ** 2 + 1) - 1

                    scores = 1 - torch.exp(-dists)
                    losses = torch.where(semi_targets == 0, dists, -torch.log(scores + self.eps))
                    loss = torch.mean(losses)

                if self.objective == 'deepSAD':
                    dists = torch.norm(outputs, p=2, dim=1) ** 2
                    scores = dists
                    losses = torch.where(semi_targets == 0, dists, ((dists + self.eps) ** semi_targets.float()))
                    loss = torch.mean(losses)

                if self.objective in ['bce', 'focal']:
                    targets = torch.zeros(inputs.size(0))
                    targets[semi_targets == -1] = 1
                    targets = targets.view(-1, 1).to(self.device)

                    scores = torch.sigmoid(outputs)
                    loss = criterion(outputs, targets)

                if epoch < self.n_epochs:
                    loss.backward()
                    optimizer.step()

                # save train scores in last epoch
                if epoch == self.n_epochs:
                    idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                                labels.cpu().data.numpy().tolist(),
                                                scores.flatten().cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

            # Take learning rate scheduler step
            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_last_lr()[0]))

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        self.train_time = time.time() - start_time
        self.train_scores = idx_label_score

        # Log results
        logger.info('Train Time: {:.3f}s'.format(self.train_time))
        logger.info('Train Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set loss
        if self.objective in ['bce', 'focal']:
            if self.objective == 'bce':
                criterion = nn.BCEWithLogitsLoss()
            if self.objective == 'focal':
                criterion = FocalLoss(gamma=self.focal_gamma)
            criterion = criterion.to(self.device)

        # Set device for network
        net = net.to(self.device)

        # Testing
        logger.info('Starting testing...')
        net.eval()
        epoch_loss = 0.0
        n_batches = 0
        idx_label_score = []
        start_time = time.time()

        with torch.no_grad():
            for data in test_loader:
                inputs, labels, semi_targets, idx = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)
                idx = idx.to(self.device)

                outputs = net(inputs)

                if self.objective == 'hsc':
                    if self.hsc_norm == 'l1':
                        dists = torch.norm(outputs, p=1, dim=1)
                    if self.hsc_norm == 'l2':
                        dists = torch.norm(outputs, p=2, dim=1)
                    if self.hsc_norm == 'l2_squared':
                        dists = torch.norm(outputs, p=2, dim=1) ** 2
                    if self.hsc_norm == 'l2_squared_linear':
                        dists = torch.sqrt(torch.norm(outputs, p=2, dim=1) ** 2 + 1) - 1

                    scores = 1 - torch.exp(-dists)
                    losses = torch.where(semi_targets == 0, dists, -torch.log(scores + self.eps))
                    loss = torch.mean(losses)

                if self.objective == 'deepSAD':
                    dists = torch.norm(outputs, p=2, dim=1) ** 2
                    scores = dists
                    losses = torch.where(semi_targets == 0, dists, ((dists + self.eps) ** semi_targets.float()))
                    loss = torch.mean(losses)

                if self.objective in ['bce', 'focal']:
                    targets = torch.zeros(inputs.size(0))
                    targets[semi_targets == -1] = 1
                    targets = targets.view(-1, 1).to(self.device)

                    scores = torch.sigmoid(outputs)
                    loss = criterion(outputs, targets)

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.flatten().cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)

        # Log results
        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Test AUC: {:.2f}'.format(100. * self.test_auc))
        logger.info('Finished testing.')
