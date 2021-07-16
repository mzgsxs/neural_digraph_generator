from __future__ import (division, print_function)
import os
import time
import pickle
from collections import defaultdict

# third party libraries
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from tensorboardX import SummaryWriter
# libraries
import models
import utils.data
from utils import data_to_gpu, snapshot, load_model, EarlyStopper, compute_edge_ratio, draw_and_save_graph
from utils.logger import get_logger


# workaround for solving the issue of multi-worker
# https://github.com/pytorch/pytorch/issues/973
try:
  import resource
  rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
  resource.setrlimit(resource.RLIMIT_NOFILE, (10000, rlimit[1]))
except:
  pass

logger = get_logger('exp_logger')

class executor(object):

  def __init__(self, config):
    #assert self.config.use_gpu == True
    self.config = config
    self.num_gpu = len(self.config.gpus)
    # setting up tensorboardx summary writer
    self.writer = SummaryWriter(config.save_dir)
    if self.config.train.resume:
      self.config.save_dir = self.config.train.resume_dir
    self.split_data()


  def split_data(self):
    # load raw graphs into networkx format
    self.graphs = utils.data.raw_graphs_to_nx(self.config.dataset, 
                                              data_dir=self.config.dataset.data_path)
    # shuffle all graphs
    if self.config.dataset.shuffle:
      self.npr = np.random.RandomState(self.config.seed)
      self.npr.shuffle(self.graphs)
    # split into train/test
    self.num_graphs = len(self.graphs)
    self.num_train = int(float(self.num_graphs) * self.config.dataset.train_ratio)
    self.num_val = int(float(self.num_train) * self.config.dataset.val_ratio)
    self.num_test = self.num_graphs - self.num_train
    logger.info('Train/val/test = {}/{}/{}'.format(self.num_train, 
                                                   self.num_val,
                                                   self.num_test))
    self.graphs_train = self.graphs[:self.num_train]
    self.graphs_test = self.graphs[self.num_train:]
    self.graphs_val = self.graphs[:self.num_val]
    # get stats of training set 
    self.config.dataset.sparse_ratio = compute_edge_ratio(self.graphs_train)  
    logger.info('No Edges vs. Edges in training set = {}'.format(compute_edge_ratio(self.graphs_train)))
    self.num_nodes_pmf_train = np.bincount([len(gg.nodes) for gg in self.graphs_train])    
    self.num_nodes_pmf_train = self.num_nodes_pmf_train / self.num_nodes_pmf_train.sum()
    self.max_num_nodes = len(self.num_nodes_pmf_train)
    # save split for benchmarking
    if self.config.dataset.save_split: 
      base_path = os.path.join(self.config.exp_dir, self.config.exp_name, 'data_split')
      if not os.path.exists(base_path): os.makedirs(base_path) 
      with open(os.path.join(base_path, '{}_train.p'.format(self.config.dataset.name)), "wb") as f:
        pickle.dump(self.graphs_test, f)
      with open(os.path.join(base_path, '{}_test.p'.format(self.config.dataset.name)), "wb") as f:
        pickle.dump(self.graphs_val, f)
      with open(os.path.join(base_path, '{}_val.p'.format(self.config.dataset.name)), "wb") as f:
        pickle.dump(self.graphs_train, f)


  def train(self):
    # create models
    model = eval('models.'+self.config.model.name)(self.config.model)
    #if self.config.gpu_data_parallel: TODO data parallel does not support sparse tensors
    #  model = torch.nn.parallel.DataParallel(model, device_ids=self.config.gpus).to(self.config.device)
    model = model.to(0)
    print(model)
    # create optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = eval('optim.'+self.config.train.optimizer)(
                            params,
                            lr=self.config.train.lr,
                            weight_decay=self.config.train.wd,
                            #momentum=self.config.train.momentum,
                            )
    early_stop = EarlyStopper([0.0], win_size=100, is_decrease=False)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
                          optimizer,
                          milestones=self.config.train.lr_decay_epoch,
                          gamma=self.config.train.lr_decay)
    # reset gradient
    optimizer.zero_grad()
    # resume training
    resume_epoch = 0
    if self.config.train.resume:
      model_path = os.path.join(self.config.train.resume_dir, self.config.train.resume_model_file_name)
      load_model(model,
                 model_path,
                 self.config.device,
                 optimizer=optimizer,
                 scheduler=lr_scheduler)
      resume_epoch = self.config.train.resume_epoch
    # load data into torch loader
    train_dataset = eval('utils.data.'+self.config.dataset.loader_name)(self.config, self.graphs_train, tag='train')
    batch_size = self.config.train.batch_size_per_gpu*self.num_gpu
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=self.config.train.shuffle,
                                               num_workers=self.config.train.num_workers,
                                               collate_fn=train_dataset.collate_fn,
                                               pin_memory=False, 
                                               drop_last=False)
    # Training Loop
    results = defaultdict(list)
    for epoch in range(resume_epoch, self.config.train.max_epoch):
      model.train()
      lr_scheduler.step()
      #train_iterator = train_loader.__iter__()
      for iter_count, (batch_graphs_gpus, _) in enumerate(train_loader):
        optimizer.zero_grad()
        # final batch size to the model is (num_batch*num_orderings*(1+num_subgraphs)) in list
        train_loss = model.loss(batch_graphs_gpus, model(batch_graphs_gpus)) 
        train_loss.backward()
        optimizer.step()
        # logging
        self.writer.add_scalar('train_loss', train_loss, iter_count)
        results['train_loss'] += [float(train_loss.data.cpu().numpy())]
        results['train_step'] += [iter_count]
        if iter_count % self.config.train.display_interval == 0: # starting from 0
          logger.info("Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count, train_loss))
      # snapshot model
      if (epoch + 1) % self.config.train.snapshot_interval == 0:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        #snapshot(model, optimizer, self.config, epoch + 1, scheduler=lr_scheduler)
        snapshot(model, optimizer, self.config, 0, scheduler=lr_scheduler) # NOTE for convinence, always use epoch 0
        # inspect the performance on the last batch
        if self.config.train.inspect is True:
          graphs = batch_graphs_gpus[0:8]
          existance_score, features = model.sample(15) # draw 16 graphs in 4 by 4
          graphs_hat = model.to_graph(existance_score, features)
          # convert sparse graphs to networkx format and draw the graph and save it
          utils.draw_a_list_of_graphs(graphs+graphs_hat, 4, 4, os.path.join(self.config.save_dir,'generation_samples'))
    # save final training stats
    pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
    self.writer.close()
    return True



  def sample(self):
    # create models
    model = eval('models.'+self.config.model.name)(self.config.model)
    model = model.to(0)
    # load model
    model_path = os.path.join(self.config.inspect.resume_dir, self.config.inspect.resume_model_file_name)
    load_model(model,
               model_path,
               self.config.device)
    # Sampling Loop
    existance_score, features = model.sample(16) # draw 16 graphs in 4 by 4
    graphs_hat = model.to_graph(existance_score, features)
    # convert sparse graphs to networkx format and draw the graph and save it
    utils.draw_a_list_of_graphs(graphs_hat, 4, 4, os.path.join(self.config.save_dir,'generation_samples'))
    return True



