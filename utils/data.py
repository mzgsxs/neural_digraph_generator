###############################################################################
#
# adapted from :
#  https://github.com/JiaxuanYou/graph-generation
#  https://github.com/lrjconan/GRAN
#
###############################################################################
import os
import time
import glob
import pickle
import torch
import torch.nn.functional as torfunc
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict


def raw_graphs_to_nx(graph_type, data_dir='data', noise=10.0, seed=1234):
  '''
  load original graph datasets from disk or generate on the fly
  then convert them into networkx formats
  '''
  npr = np.random.RandomState(seed)
  graphs = []
  # generate digraphs
  if graph_type == 'digraph':
    graphs = []
    for i in range(1024):
      G = nx.gn_graph(np.random.randint(64,128))
      G = nx.stochastic_graph(G)
      graphs.append(G)    
  # load local graph data TODO
  # Print info for the data set
  num_nodes = [gg.number_of_nodes() for gg in graphs]
  num_edges = [gg.number_of_edges() for gg in graphs]
  print('max # nodes = {} || mean # nodes = {}'.format(max(num_nodes), np.mean(num_nodes)))
  print('max # edges = {} || mean # edges = {}'.format(max(num_edges), np.mean(num_edges))) 
  return graphs


class digraph_loader(object):
  '''
  Load digraph-data of networkx format, save them into files in the disk
  When fetched, indices are used to get data from saved files
  A batch of indices are buddled into a torch tensors with outer dimension(left-most) batch_size
  '''
  def __init__(self, config, graphs, tag='train'):
    self.config = config
    self.tag = tag
    self.graphs = graphs
    self.num_graphs = len(graphs)
    self.npr = np.random.RandomState(self.config.seed)
    self.num_orderings = len(self.config.dataset.node_orderings)
    # more detailed parameters
    self.stride = config.dataset.stride
    self.block_size = config.dataset.block_size
    if self.config.dataset.sample_subgraph:
      assert self.config.dataset.num_subgraphs_to_sample > 0
    # pre-process the data and save it to disk
    self.save_path = os.path.join(self.config.exp_dir, 
           'data_pre_processed','{}_{}_block_size_{}_stride_{}'.format(
            config.dataset.name, tag, self.block_size, self.stride, 
            self.config.dataset.node_orderings))
    if not os.path.isdir(self.save_path) or self.config.dataset.over_write_pre_saved:
      self.file_names = []
      if not os.path.isdir(self.save_path):
        os.makedirs(self.save_path)
      self.config.dataset.save_path = self.save_path
      for index in tqdm(range(self.num_graphs)):
        G = self.graphs[index]
        data = self._get_graph_data(G)
        tmp_path = os.path.join(self.save_path, '{}_{}.p'.format(tag, index))
        pickle.dump(data, open(tmp_path, 'wb'))
        self.file_names += [tmp_path]
    else:
      self.file_names = glob.glob(os.path.join(self.save_path, '*.p'))


  def __len__(self):
    return self.num_graphs


  def _get_graph_data(self, G):
    '''
    load a single networkx formated graph and outputs a list of all orderings
    In:
      G: single networkx graph structure
    Output:
      adjs: [Adj_order1, Adj_order2, ...]  
    '''
    node_degree_list = [(n, d) for n, d in G.degree()]
    adj = {}
    adj['original'] = np.array(nx.to_numpy_matrix(G))
    ### Degree descent ranking
    if 'degree_descent' in self.config.dataset.node_orderings:
      # N.B.: largest-degree node may not be unique
      degree_sequence = sorted(
          node_degree_list, key=lambda tt: tt[1], reverse=True)
      adj['degree_descent'] = np.array(
          nx.to_numpy_matrix(G, nodelist=[dd[0] for dd in degree_sequence]))
    ### Degree ascent ranking
    if 'degree_ascent' in self.config.dataset.node_orderings:
      degree_sequence = sorted(node_degree_list, key=lambda tt: tt[1])
      adj['degree_ascent'] = np.array(
          nx.to_numpy_matrix(G, nodelist=[dd[0] for dd in degree_sequence]))
    ### K-core ranking
    if 'k-core' in self.config.dataset.node_orderings:
      num_core = nx.core_number(G)
      core_order_list = sorted(list(set(num_core.values())), reverse=True)
      degree_dict = dict(G.degree())
      core_to_node = defaultdict(list)
      for nn, kk in num_core.items():
        core_to_node[kk] += [nn]
      node_list = []
      for kk in core_order_list:
        sort_node_tuple = sorted(
            [(nn, degree_dict[nn]) for nn in core_to_node[kk]],
            key=lambda tt: tt[1],
            reverse=True)
        node_list += [nn for nn, dd in sort_node_tuple]
      adj['k-core'] = np.array(nx.to_numpy_matrix(G, nodelist=node_list))
    ### fill in required orderings
    adjs = [adj[order_name] for order_name in self.config.dataset.node_orderings] 
    return adjs


  def __getitem__(self, index):
    '''
    get a single graph and create subgraph samples for it 
    In:
      index: w.r.t the dataset
    Out:
      sub_graphs: tensor 
        shape: num_all_subgraphs x N_max x N_max x dim_edge_feature
          num_all_subgraphs = num_orderings*(1+num_subgraphs)
          num_subgraphs may vary for different graphs
      sizes: list
        shape: num_all_subgraphs x 1
    '''
    K, S = self.block_size, self.stride
    # load one single graph
    adj_list = pickle.load(open(self.file_names[index], 'rb'))
    num_nodes = adj_list[0].shape[0]
    subgraphs, sizes = [], []
    if self.config.dataset.use_subgraphs:
      # use all sub-graphs possible
      subgraph_indices = np.arange(0,num_nodes-K,S)
      num_subgraphs = len(subgraph_indices)
      # if sample, only use part of all subgrahs
      if self.config.dataset.sample_subgraph:
        num_subgraphs = min(num_subgraphs, self.config.dataset.num_subgraphs_to_sample)
        subgraph_indices = self.npr.randint(0,high=(num_nodes-K),size=num_subgraphs).tolist()
    # for each ordering
    for i in range(len(adj_list)):
      adj_order_i = adj_list[i]
      # makesure its a 3 dimensional array, i.e. num_nodes x num_nodes x dim_edge_feature
      if adj_order_i.ndim == 2: adj_order_i = np.expand_dims(adj_order_i,-1)
      adj_order_i_tor = torch.from_numpy(adj_order_i.astype('float32'))
      # the last dimension is dense
      adj_order_i_tor = adj_order_i_tor.to_sparse(2)
      # make sure the full graph is always included
      subgraphs.append(adj_order_i_tor)
      sizes.append(num_nodes)
      if self.config.dataset.use_subgraphs:
        for j in subgraph_indices:
          # get the subgraph
          adj_oi_subgraph_j = adj_order_i[:j, :j, :]
          adj_subgraph_tor = torch.from_numpy(adj_oi_subgraph_j.astype('float32'))
          adj_subgraph_tor = adj_subgraph_tor.to_sparse(2)
          subgraphs.append(adj_subgraph_tor)
          sizes.append(j)
    return (subgraphs, sizes)


  def collate_fn(self, batch_data):
    '''
    takes a list of batch_size numbers of data, each data in the format of __getitem__(),
    combine batch_size dimension and num_subgraphs dimension and load all data into gpu
      batch_size = (batch_size_per_gpu * num_gpu)
    In:
      batch_data: list of tuple (tensors, sizes)
        shape: [batch_size x (num_all_subgraphs x N_max x N_max, num_all_subgraphs x 1)]
    Output:
      batch_graphs_gpus: list of tensors loaded in gpus
        shape: [(batch_size * num_all_subgraphs) x (N_max x N_max x dim_edge_feature)]
      sizes: list of int
        shape: [(batch_size * num_all_subgraphs) x 1]
    '''
    assert isinstance(batch_data, list)
    batch_size = len(batch_data)
    batch_graphs_gpus = []
    sizes = []
    for gpu_id in self.config.gpus:
      for data in batch_data[gpu_id*self.config.train.batch_size_per_gpu:
                            (gpu_id+1)*self.config.train.batch_size_per_gpu]:
        batch_graphs_gpus += [g.to(gpu_id, non_blocking=True) for g in data[0]] 
        sizes += data[1]
    return (batch_graphs_gpus, sizes)

