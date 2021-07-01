import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphNodeEmbedding(nn.Module):
  '''
  Graph node embeddding module for directed graph
    node states and directed edge feature are used to generate both the deaft message and attention weights.
    draft message and attention weights are multiplied by a dot product to produce the final message.
    all final message to a node is sumed, this message sum is input to a recurrent module to update the node state.
  Params:
    dim_message: message vector length
    dim_node_state: node state vector length
    dim_edge_feature: edge feature vector length
    dim_attention_hidden_layer: number of neurons in the hidden layer for 
    num_layer: number of weights for message passing 
    num_message_passing: number of message passing for the same set of weights 
  '''
  def __init__(self,dim_node_feature,
                    dim_edge_feature,
                    dim_message,
                    dim_node_state,
                    dim_hidden_input,
                    dim_hidden_readout,
                    dim_hidden_attention,
                    dim_readout,
                    num_layer,
                    num_message_passing):
    super(GraphNodeEmbedding, self).__init__()
    self.dim_node_feature      =  dim_node_feature
    self.dim_edge_feature      =  dim_edge_feature
    self.dim_message           =  dim_message
    self.dim_node_state        =  dim_node_state
    self.dim_hidden_input      =  dim_hidden_input
    self.dim_hidden_readout    =  dim_hidden_readout
    self.dim_hidden_attention  =  dim_hidden_attention
    self.dim_readout           =  dim_readout
    self.num_layer             =  num_layer
    self.num_message_passing   =  num_message_passing
    # computation components
    self.input_func = nn.Sequential(
            *[nn.Linear(self.dim_node_feature, self.dim_hidden_input),
              nn.ReLU(),
              nn.Linear(self.dim_hidden_input, self.dim_node_state)])
    self.message_func = nn.ModuleList([nn.Sequential(
            *[nn.Linear(2*self.dim_node_state + self.dim_edge_feature,
                        self.dim_message),
              nn.ReLU(),
              nn.Linear(self.dim_message, self.dim_message)
            ]) for _ in range(self.num_layer)])
    self.attention_weight = nn.ModuleList([nn.Sequential(
            *[nn.Linear(2*self.dim_node_state + self.dim_edge_feature,
                        self.dim_hidden_attention),
              nn.ReLU(),
              nn.Linear(self.dim_hidden_attention, self.dim_message),
              nn.Sigmoid()
            ]) for _ in range(self.num_layer)])
    self.residual_connection = nn.ModuleList([nn.Sequential(
            *[nn.Linear(self.dim_node_state*2,self.dim_node_state)
            ]) for _ in range(self.num_layer)])
    self.update_func = nn.ModuleList([
              nn.GRUCell(input_size=self.dim_message, hidden_size=self.dim_node_state)
              for _ in range(self.num_layer)])
    self.readout_func = nn.Sequential(
            *[nn.Linear(self.dim_node_state, self.dim_hidden_readout),
              nn.ReLU(),
              nn.Linear(self.dim_hidden_readout, self.dim_readout)])


  def _message_passing(self, nodes_state, edges, edges_feature, layer_idx):
    '''
    update node states once
    In:
      nodes_state: num_nodes x dim_node_feature
      edges: num_edges x 2, [source, targets]
      edges_features: num_edges x dim_edge_feature
    Out:
      new_nodes_state: shape same as nodes_state
    '''
    # num_edges x (2*dim_node_feature + dim_edge_feature)
    edge_input = torch.cat([nodes_state[edges[:, 0], :], nodes_state[edges[:, 1], :], edges_feature], dim=1)
    # both of shape: num_edges x dim_message
    msg = self.message_func[layer_idx](edge_input)*self.attention_weight[layer_idx](edge_input)
    # sum all message for each node,  num_nodes x dim_message
    msg_sum = torch.zeros(nodes_state.shape[0], self.dim_message, dtype=msg.dtype, device=msg.device
                          ).scatter_add(0, edges[:, [1]].expand(-1, self.dim_message), msg)
    # post summation integrate 
    new_nodes_state = self.update_func[layer_idx](msg_sum, nodes_state)
    return new_nodes_state


  def forward(self, nodes_feature, edges, edges_feature):
    """
    A batch of graphs can be seen as one large graph with isolated sub-graphs
    In:
      nodes_feature: num_nodes X dim_node_feature
      edges: num_edges X 2, [:,0] source, [:,1] target
      edges_feature: num_edges X dim_edge_feature
    Out:
      nodes_state: num_nodes x dim_node_state
    """
    # initial nodes' state
    nodes_state = self.input_func(nodes_feature)
    for i in range(self.num_layer):
      old_nodes_state = nodes_state
      if i > 0: nodes_state = F.relu(nodes_state)
      for j in range(self.num_message_passing):
        nodes_state = self._message_passing(nodes_state, edges, edges_feature, i)
      if self.residual_connection:
        nodes_state = self.residual_connection[i](torch.cat((nodes_state, old_nodes_state), dim=-1))
    return self.readout_func(nodes_state)




class GraphSumPool(nn.Module):
  '''
  Graph sum pooling module for directed graph
  Params:
  '''
  def __init__(self,dim_node_embedding,
                    dim_hidden_graph_readout,
                    dim_graph_embedding):
    super(GraphSumPool, self).__init__()
    self.graph_readout_func = nn.Sequential(
            *[nn.Linear(dim_node_embedding, dim_hidden_graph_readout),
              nn.ReLU(),
              nn.Linear(dim_hidden_graph_readout, dim_graph_embedding)])


  def forward(self, nodes_embedding, graphs_size):
    """
    A batch of graphs 
    In:
      nodes_embedding: num_nodes x dim_node_embedding
      graphs_size: num_graphs x 1
    Out:
      graphs_embedding: num_graphs x dim_graph_embedding
    """
    graphs_size = torch.tensor(graphs_size, device=nodes_embedding.device)
    graph_idx = torch.arange(0,graphs_size.shape[0], device=nodes_embedding.device).repeat_interleave(graphs_size)
    # TODO data parallel will not work for now
    nodes_sum = torch.zeros((graphs_size.shape[0], nodes_embedding.shape[1]), 
                            dtype=nodes_embedding.dtype, device=nodes_embedding.device
                            ).scatter_add(0, graph_idx.unsqueeze(-1).expand(-1, nodes_embedding.shape[1]),
                              nodes_embedding)
    graphs_embedding = self.graph_readout_func(nodes_sum)
    return graphs_embedding



class GraphGenRnn(nn.Module):
  '''
  Graph generator based on recurrent neural network
  Params:
  '''
  def __init__(self,dim_gene_code,
                    dim_edge_feature,
                    dim_lstm_node,
                    dim_lstm_edge):
    super(GraphGenRnn, self).__init__()
    self.dim_lstm_node = dim_lstm_node
    self.dim_lstm_edge = dim_lstm_edge
    self.dim_edge_feature = dim_edge_feature
    self.node_rnn = nn.LSTMCell(dim_gene_code, dim_lstm_node)
    self.edge_rnn = nn.LSTMCell(dim_gene_code+256, dim_lstm_edge)
    self.node_entry_func = nn.Sequential(
            *[nn.Linear(dim_lstm_node,256),
              nn.ReLU(),
              nn.Linear(256, 256)])
    self.edge_entry_func = nn.Sequential(
            *[nn.Linear(dim_lstm_edge, 512),
              nn.ReLU(),
              nn.Linear(512, 256),
              nn.ReLU(),
              nn.Linear(256, 256)])
    self.edge_existance_score = nn.Sequential(
            *[nn.Linear(256, 128),
              nn.ReLU(),
              nn.Linear(128, 2),
              nn.Sigmoid()])
    self.edge_feature = nn.Sequential(
            *[nn.Linear(256, 128),
              nn.ReLU(),
              nn.Linear(128, 2*dim_edge_feature)])


  def forward(self, gene_codes, num_nodes_max):
    """
    generate a batch of graphs 
    In:
      gene_codes: tensor
        shape: num_graphs x dim_gene_codes
      num_nodes_max: int
    Out:
      existance_score: tensor 
        shape: num_graphs x N_max x N_max x 2
      features: tensor
        shape: num_graphs x N_max x N_max x dim_edge_feature
    """
    num_graphs = gene_codes.shape[0]
    hx_node = torch.randn(num_graphs, self.dim_lstm_node, device=gene_codes.device)
    cx_node = torch.randn(num_graphs, self.dim_lstm_node, device=gene_codes.device)
    edges = []
    existance_score = torch.zeros(num_graphs, num_nodes_max, num_nodes_max, 1, device=gene_codes.device)
    features = torch.zeros(num_graphs, num_nodes_max, num_nodes_max, self.dim_edge_feature, device=gene_codes.device)
    for node_i in range(2, num_nodes_max):
      # num_graphs x dim_lstm_node
      hx_node, cx_node = self.node_rnn(gene_codes, (hx_node, cx_node)) 
      node_entry_point = self.node_entry_func(hx_node)
      # num_graphs x (dim_gene_node + dim_node_entry)
      node_level_gene = torch.cat([gene_codes, node_entry_point], dim=1) 
      # random initial lstm states
      hx_edge = torch.randn(num_graphs, self.dim_lstm_edge, device=gene_codes.device)
      cx_edge = torch.randn(num_graphs, self.dim_lstm_edge, device=gene_codes.device)
      edge_existance_prob, edge_feature = [], []
      # ignore digonal entry, no self connection
      for edge_i in range(node_i-1):
        hx_edge, cx_edge = self.edge_rnn(node_level_gene, (hx_edge, cx_edge))
        edge_entry_point = self.edge_entry_func(hx_edge)
        # num_graphs x 2 -> num_graphs x 2 x 1
        edge_existance_prob.append(torch.reshape(self.edge_existance_score(edge_entry_point), 
                                                 (num_graphs, 2, 1)))
        # num_graphs x (2*dim_edge_feature) -> num_graphs x 2 x dim_edge_feature
        edge_feature.append(torch.reshape(self.edge_feature(edge_entry_point), 
                                          (num_graphs, 2, self.dim_edge_feature)))
      edge_existance_prob_layer = torch.stack(edge_existance_prob, dim=1)
      edge_feature_layer = torch.stack(edge_feature, dim=1)
      # put layers of edges into matrix
      existance_score[:, node_i, :(node_i-1), :] = edge_existance_prob_layer[:,:,0,:]
      existance_score[:, :(node_i-1), node_i, :] = edge_existance_prob_layer[:,:,1,:]
      features[:, node_i, :(node_i-1), :] = edge_feature_layer[:,:,0,:]
      features[:, :(node_i-1), node_i, :] = edge_feature_layer[:,:,1,:]
    return existance_score, features

 

#TODO finish the implementation
class GraphClusteringPooling(nn.Module):
  ''' 
  Graph pooling module for directed graph via clustering
  Params:
    num_nodes: of the input graph
    shrink_ratio: w.r.t. the num_nodes
  '''
  def __init__(self, num_nodes, shrink_ratio):
    self.score_fun = nn.Sequential(
                      GraphNodeEmbedding(dim_state = ceil(shrink_fraction*num_nodes)),
                      nn.Softmax(dim=-1))
    self.num_nodes = num_nodes

  def forward(self, nodes_feature, edges, edges_feature):
    """
    A batch of graphs can be seen as one large graph with isolated sub-graphs
    In:
      nodes_feature: num_nodes X dim_node_feature
      edges: num_edges X 2, [:,0] source, [:,1] target
      edges_feature: num_edges X dim_edge_feature
    Out:
      nodes_feature_new: num_nodes_new x dim_node_state
      edges_feature_new: num_edges_new x dim_edge_state
      edges_new: num_edges_new x 2, [:,0] source, [:,1] target
    """
    # num_nodes x (num_nodes*shrink_ratio)
    score = self.score_func(nodes_feature, edges, edges_feature)
    score_transpose = torch.transpose(score,-1,-2) 
    nodes_feature_new = torch.matmul(score, nodes_feature)
    adjacency_tensor = torch.sparse_coo_tensor(edges, edges_feature,
                              size=(self.num_nodes, self.num_nodes, dim_edge_feature))
    adjacency_tensor_new = torch.matmul(torch.matmul(score_transpose, adjacency_tensor), score)
    return nodes_feature_new, adjacency_tensor_new.indices(), adjacency_tensor_new.values()



# TODO
class GraphEdgeEmbedding(nn.Module):
  '''
  Graph edge embedding module for directed graph
  Params:
  '''
  def __init__(self):
    pass

  def forward(self, nodes_feature, edges, edges_feature):
    """
    A batch of graphs can be seen as one large graph with isolated sub-graphs
    In:
      nodes_feature: num_nodes X dim_node_feature
      edges: num_edges X 2, [:,0] source, [:,1] target
      edges_feature: num_edges X dim_edge_feature
    Out:
      edges_state: num_edges x dim_edge_state
    """
    return edges_state




  
