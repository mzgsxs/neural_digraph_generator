import torch
import torch.nn as nn
import torch.nn.functional as F
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
    # encoder modules
    self.encoder_gnn = models.GraphNodeEmbedding( dim_node_feature      =  1     ,
                                                  dim_edge_feature      =  1     ,
                                                  dim_message           =  128   ,
                                                  dim_node_state        =  128   ,
                                                  dim_hidden_input      =  64    ,
                                                  dim_hidden_readout    =  128   ,
                                                  dim_hidden_attention  =  128   ,
                                                  dim_readout           =  128   ,
                                                  num_layer             =  1     ,
                                                  num_message_passing   =  1     ,
                                                 )
    self.encoder_pooling = models.GraphSumPool( dim_node_embedding        =  128  ,
                                                dim_hidden_graph_readout  =  128  ,
                                                dim_graph_embedding       =  128  ,
                                               )
    # decoder modules
    self.decoder_graph_rnn = models.GraphGenRnn( dim_gene_code     = 128  ,
                                                 dim_edge_feature  = 1    , 
                                                 dim_lstm_node     = 256  ,
                                                 dim_lstm_edge     = 256  ,
                                                )


  def encoder(self, nodes_feature, indices, values, batch_graphs_size):
    '''
    In:
      nodes_feature, indices, values 
        are outputs from _graph_sparse_tensors_to_idxs_and_vals 
      batch_graphs_size: tensor
        shape: num_graphs x 1
    Out:
      gene_code: sparse tensor
        shape: num_graphs x dim_gene_code
    '''
    nodes_embedding = self.encoder_gnn(nodes_feature, indices, values)
    gene_codes = self.encoder_pooling(nodes_embedding, batch_graphs_size)
    return gene_codes 


  # TODO generate arbitary sized graph
  def decoder(self, gene_codes):
    '''
    In:
      gene_codes: tensor
        shape: num_graphs x dim_gene_codes
    Out:
      existance_score: tensor 
        shape: num_graphs x self.num_nodes_max x self.num_nodes_max x 1
      features: tensor
        shape: num_graphs x self.num_nodes_max x self.num_nodes_max x dim_edge_feature
    ''' 
    existance_score, features = self.decoder_graph_rnn(gene_codes, self.num_nodes_max)
    return existance_score, features


  def _to_graph(self):
    return graphs


  def _graph_sparse_tensors_to_idxs_and_vals(self, batch_graphs): 
    '''
    Gte a num_graphs of graphs and return their indices and values,
    In:
      batch_graphs: sparse tensor
        shape: num_graphs x [num_nodes x num_nodes x dim_edge_feature]
        num_nodes is different for each graph
    Out:
      nodes_feature: tensor
        shape: num_nodes x 1
      indices: tensor
        shape: num_edges x 2
      values: tensor
        shape: num_edges x dim_edge_feature
      batch_graphs_size: list of int
        shape: num_graphs x 1
    '''
    indices,values,nodes_feature,batch_graphs_size = [],[],[],[]
    starting_index = 0
    for i, g in enumerate(batch_graphs): 
      g_size = g.shape[0]
      batch_graphs_size.append(g_size)
      indices.append(g.indices().transpose(0,1) + starting_index)
      values.append((g.values()))
      nodes_feature.append(torch.ones((g_size,1), device=g.get_device()))
      # graph size is added per iteration to move the starting index for the next graph
      starting_index += g_size 
    self.num_nodes_max = max(batch_graphs_size)
    return torch.cat(nodes_feature,dim=0), torch.cat(indices,dim=0), torch.cat(values,dim=0), batch_graphs_size


  def forward(self, batch_graphs, batch_graphs_size):
    '''
    Batch size after data augumentation is:
      num_graphs or actual_batch_size = bacth_size*num_orderings*(1+num_subgraphs)
      see more in data.py
    In:
      batch_graphs: list of sparse tensors
        shape: num_graphs x [N_max x N_max x dim_edge_feature]
      batch_graphs_size: list of int
        shape: num_graphs x 1
    Out:
      same as outputs from decoders
    '''
    nodes_feature, indices, values, batch_graphs_size = self._graph_sparse_tensors_to_idxs_and_vals(batch_graphs)
    gene_codes = self.encoder(nodes_feature, indices, values, batch_graphs_size)
    existance_score, features = self.decoder(gene_codes)
    return existance_score, features


  def loss(self, batch_graphs, existance_score, features):
    '''
    In:
      batch_graphs: list of sparse tensors
        shape: num_graphs x [num_nodes x num_nodes x dim_edge_feature]
      existance_score: tensor 
        shape: num_graphs x self.num_nodes_max x num_nodes_max x 1
      features: tensor
        shape: num_graphs x self.num_nodes_max x num_nodes_max x dim_edge_feature
    Out:
      loss: scalar 
    '''
    starting_index = 0
    # resize all for indexing
    all_graphs = []
    dim_edge_feature = batch_graphs[0].shape[-1]
    for i, g in enumerate(batch_graphs): 
      g.sparse_resize_((self.num_nodes_max, self.num_nodes_max, dim_edge_feature), 2, 1)
      all_graphs.append(g)
    all_graphs = torch.stack(all_graphs, dim=0).coalesce()
    indices = all_graphs.indices()
    values = all_graphs.values()
    # num_edges x 1   ->   scalar
    existance_target = torch.sparse_coo_tensor(indices,
                             torch.ones_like(values, device=existance_score.device),
                             existance_score.shape).to_dense()
    features_val = features[indices[0,:],indices[1,:],indices[2,:],:]
    # loss
    ce_loss = F.binary_cross_entropy(existance_score, existance_target)
    feature_loss = F.mse_loss(features_val, values)
    return ce_loss+feature_loss



