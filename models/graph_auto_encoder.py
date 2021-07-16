import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import models


class graph_auto_encoder(nn.Module):
  '''
  """ Graph auto encoder """
  Auto-encoder structure based on graph neural nets
    decoder can be used for stand alone graph distribution learning 
    with gene_codes generated from a pre-specified distribution
  '''
  def __init__(self, config_model):
    super(graph_auto_encoder, self).__init__()
    self.config = config_model
    self.decoder_graph_gnn = models.GraphGenGnn(self.config)


  def _graph_sparse_tensors_to_idxs_and_vals(self, graphs): 
    '''
    Gte a num_graphs of graphs and return their indices and values,
    In:
      graphs: sparse tensor
        shape: num_graphs x [num_nodes x num_nodes x dim_edge_feature]
        num_nodes is different for each graph
    Out:
      nodes_feature: tensor
        shape: num_nodes x 1
      indices: tensor
        shape: num_edges x 2
      values: tensor
        shape: num_edges x dim_edge_feature
      graphs_size: list of int
        shape: num_graphs x 1
    '''
    indices,values,nodes_feature,graphs_size = [],[],[],[]
    starting_index = 0
    for i, g in enumerate(graphs): 
      g_size = g.shape[0]
      graphs_size.append(g_size)
      indices.append(g.indices().transpose(0,1) + starting_index)
      values.append((g.values()))
      nodes_feature.append(torch.ones((g_size,1), device=g.get_device()))
      # graph size is added per iteration to move the starting index for the next graph
      starting_index += g_size 
    return torch.cat(nodes_feature,dim=0), torch.cat(indices,dim=0), torch.cat(values,dim=0), graphs_size


  def encoder(self, nodes_feature, indices, values, graphs_size):
    '''
    In:
      nodes_feature, indices, values 
        are outputs from _graph_sparse_tensors_to_idxs_and_vals 
      graphs_size: tensor
        shape: num_graphs x 1
    Out:
      gene_code: sparse tensor
        shape: num_graphs x dim_gene_code
    '''
    nodes_embedding = self.encoder_gnn(nodes_feature, indices, values)
    gene_codes = self.encoder_pooling(nodes_embedding, graphs_size)
    return gene_codes 


  def decoder(self, gene_codes, graphs_size):
    '''
    In:
      gene_codes: tensor
        shape: num_graphs x dim_gene_codes
    Out:
      existance_score: tensor 
        shape: num_graphs x self.num_nodes_max x self.num_nodes_max
      features: tensor
        shape: num_graphs x self.num_nodes_max x self.num_nodes_max x dim_edge_feature
    ''' 
    existance_score, features = self.decoder_graph_gnn(gene_codes, graphs_size)
    return existance_score, features


  def forward(self, graphs):
    '''
    Batch size after data augumentation is:
      num_graphs or actual_batch_size = bacth_size*num_orderings*(1+num_subgraphs)
      see more in data.py
    In:
      graphs: list of sparse tensors
        shape: num_graphs x [num_nodes x num_nodes x dim_edge_feature]
             * num_nodes is different for each graph
    Out:
      same as outputs from decoders
    '''
    return self.decoder_graph_gnn(graphs)


  def sample(self, num_samples):
    return self.decoder_graph_gnn._forward_sampling(None, [0]*num_samples)


  def loss(self, _, prob):
    (log_theta, log_alpha) = prob
    return self.decoder_graph_gnn._loss(log_theta, log_alpha, 
             nn.BCEWithLogitsLoss(reduction='none'))


  def to_graph(self, existance_scores, features, dense=True):
    '''
    using scores to generate graphs
    In:
      existance_score: sparse tensor 
        shape: num_graphs x num_nodes_max x num_nodes_max
      features: sparse tensor
        shape: num_graphs x num_nodes_max x num_nodes_max x dim_edge_feature
    Out: 
      graphs: list of sparse tensors, last dimension is dense
        shape: num_graphs x [num_nodes x num_nodes x dim_edge_feature]
        * num_nodes is different between different graphs
    '''
    if not dense: existance_score = existance_score.to_dense()
    edges = D.bernoulli.Bernoulli(existance_scores).sample()
    features = edges.unsqueeze(-1) # TODO for now, just use 1 for features if edge exist 
    dim_edge_feature = features.shape[-1]
    graphs = []
    for i in range(features.shape[0]):
      edges_sparse_i = edges[i,:,:].to_sparse().coalesce()
      indices = edges_sparse_i.indices()[0:2,:] # 2 x num_edges
      values = features[i,indices[0,:],indices[1,:],:]  # num_edges x dim_edge_feature 
      # only if it's non-empty graph
      if indices.shape[1]>0:
        num_nodes = torch.max(indices) + 1
        graphs.append(torch.sparse_coo_tensor(indices, values,
                              (num_nodes, num_nodes, dim_edge_feature)).coalesce())
    return graphs


  # TODO implement ordered loss, out of all orderings, only one is most close to the original
  def _loss(self, graphs, graphs_prob, dense=True):
    '''
    one extra dimension is added to indicate the end of generation
    num_nodes_max is actually  graphs_size_max + 1
    In:
      graphs: list of sparse tensors
        shape: num_graphs x [num_nodes x num_nodes x dim_edge_feature]
             * num_nodes is different for each graph
      existance_score: tensor (maybe dense) 
        shape: num_graphs x num_nodes_max x num_nodes_max
      features: tensor (maybe dense) 
        shape: num_graphs x num_nodes_max x num_nodes_max x dim_edge_feature
    Out:
      loss: scalar 
    '''
    # resize all original graphs for indexing
    (existance_score, features) = graphs_prob
    if not dense: existance_score = existance_score.to_dense()
    graphs_resized = []
    num_nodes_max = existance_score.shape[1]
    dim_edge_feature = graphs[0].shape[-1]
    for i, g in enumerate(graphs):
      g.sparse_resize_((num_nodes_max, num_nodes_max, dim_edge_feature), 2, 1)
      graphs_resized.append(g)
    graphs_resized = torch.stack(graphs_resized, dim=0).coalesce()
    edges_idx = graphs_resized.indices()
    # existance loss
    existance_target = torch.sparse_coo_tensor(edges_idx,
                               torch.ones(edges_idx.shape[-1], device=existance_score.device),
                               existance_score.shape).to_dense()
    ce_loss = F.binary_cross_entropy(existance_score, existance_target, reduction='mean')
    # feature loss #TODO ignore weight first 
    #features_val = features[indices[0,:],indices[1,:],indices[2,:],:]
    #feature_loss = F.mse_loss(features_val, values)
    return ce_loss#+feature_loss


