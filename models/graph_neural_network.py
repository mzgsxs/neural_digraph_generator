import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
  '''original graph neural network implementation from gran'''
  def __init__(self,
               msg_dim,
               node_state_dim,
               edge_feat_dim,
               num_prop=1,
               num_layer=1,
               has_attention=True,
               att_hidden_dim=128,
               has_residual=False,
               has_graph_output=False,
               output_hidden_dim=128,
               graph_output_dim=None):
    super(GNN, self).__init__()
    self.msg_dim = msg_dim
    self.node_state_dim = node_state_dim
    self.edge_feat_dim = edge_feat_dim
    self.num_prop = num_prop
    self.num_layer = num_layer
    self.has_attention = has_attention
    self.has_residual = has_residual
    self.att_hidden_dim = att_hidden_dim
    self.has_graph_output = has_graph_output
    self.output_hidden_dim = output_hidden_dim
    self.graph_output_dim = graph_output_dim

    self.update_func = nn.ModuleList([
        nn.GRUCell(input_size=self.msg_dim, hidden_size=self.node_state_dim)
        for _ in range(self.num_layer)
    ])

    self.msg_func = nn.ModuleList([
        nn.Sequential(
            *[
                nn.Linear(self.node_state_dim + self.edge_feat_dim,
                          self.msg_dim),
                nn.ReLU(),
                nn.Linear(self.msg_dim, self.msg_dim)
            ]) for _ in range(self.num_layer)
    ])

    if self.has_attention:
      self.att_head = nn.ModuleList([
          nn.Sequential(
              *[
                  nn.Linear(self.node_state_dim + self.edge_feat_dim,
                            self.att_hidden_dim),
                  nn.ReLU(),
                  nn.Linear(self.att_hidden_dim, self.msg_dim),
                  nn.Sigmoid()
              ]) for _ in range(self.num_layer)
      ])

    if self.has_graph_output:
      self.graph_output_head_att = nn.Sequential(*[
          nn.Linear(self.node_state_dim, self.output_hidden_dim),
          nn.ReLU(),
          nn.Linear(self.output_hidden_dim, 1),
          nn.Sigmoid()
      ])

      self.graph_output_head = nn.Sequential(
          *[nn.Linear(self.node_state_dim, self.graph_output_dim)])

  def _prop(self, state, edge, edge_feat, layer_idx=0):
    ### compute message
    state_diff = state[edge[:, 0], :] - state[edge[:, 1], :]
    if self.edge_feat_dim > 0:
      edge_input = torch.cat([state_diff, edge_feat], dim=1)
    else:
      edge_input = state_diff

    msg = self.msg_func[layer_idx](edge_input)

    ### attention on messages
    if self.has_attention:
      att_weight = self.att_head[layer_idx](edge_input)
      msg = msg * att_weight

    ### aggregate message by sum
    state_msg = torch.zeros(state.shape[0], msg.shape[1]).to(state.device)
    scatter_idx = edge[:, [1]].expand(-1, msg.shape[1])
    state_msg = state_msg.scatter_add(0, scatter_idx, msg)

    ### state update
    state = self.update_func[layer_idx](state_msg, state)
    return state


  def forward(self, node_feat, edge, edge_feat, graph_idx=None):
    """
      N.B.: merge a batch of graphs as a single graph

      node_feat: N X D, node feature
      edge: M X 2, edge indices
      edge_feat: M X D', edge feature
      graph_idx: N X 1, graph indices
    """

    state = node_feat
    prev_state = state
    for ii in range(self.num_layer):
      if ii > 0:
        state = F.relu(state)

      for jj in range(self.num_prop):
        state = self._prop(state, edge, edge_feat=edge_feat, layer_idx=ii)

    if self.has_residual:
      state = state + prev_state

    if self.has_graph_output:
      num_graph = graph_idx.max() + 1
      node_att_weight = self.graph_output_head_att(state)
      node_output = self.graph_output_head(state)

      # weighted average
      reduce_output = torch.zeros(num_graph,
                                  node_output.shape[1]).to(node_feat.device)
      reduce_output = reduce_output.scatter_add(0,
                                                graph_idx.unsqueeze(1).expand(
                                                    -1, node_output.shape[1]),
                                                node_output * node_att_weight)

      const = torch.zeros(num_graph).to(node_feat.device)
      const = const.scatter_add(
          0, graph_idx, torch.ones(node_output.shape[0]).to(node_feat.device))

      reduce_output = reduce_output / const.view(-1, 1)

      return reduce_output
    else:
      return state








#TODO validate this new implementatin works
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
            *[nn.Linear(self.dim_node_state + self.dim_edge_feature,
                        self.dim_message),
              nn.ReLU(),
              nn.Linear(self.dim_message, self.dim_message)
            ]) for _ in range(self.num_layer)])
    self.attention_weight = nn.ModuleList([nn.Sequential(
            *[nn.Linear(self.dim_node_state + self.dim_edge_feature,
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
    #edge_input = torch.cat([nodes_state[edges[:, 0], :], nodes_state[edges[:, 1], :], edges_feature], dim=1)
    edge_input = torch.cat([nodes_state[edges[:, 0], :] - nodes_state[edges[:, 1], :], edges_feature], dim=1)
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








class GraphGenGnn(nn.Module):
  '''
  Graph generator based on graph neural network
  Params:
  '''
  def __init__(self, config):
    super(GraphGenGnn, self).__init__()
    self.training = config.training
    self.cf = config.decoder
    # for consistency
    self.cf.dim_att_edge = self.cf.num_new_nodes + 1
    self.cf.dim_edge_feature = 2*self.cf.dim_att_edge 
    # overload forward function
    self.forward = self._forward_training if self.training else self._forward_sampling
    # computing modules
    self.adj_vecs_to_nodes_state = nn.Sequential(
        nn.Linear(2*self.cf.max_num_nodes, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, self.cf.dim_node_feature))
    self.gne = GNN(
        msg_dim=128,
        node_state_dim=self.cf.dim_node_feature,
        edge_feat_dim=self.cf.dim_edge_feature,
        num_prop=1,
        num_layer=7,
        has_attention=True)
    self.output_theta = nn.Sequential(
        nn.Linear(self.cf.dim_node_feature, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 2*self.cf.num_mix_component))
    self.output_alpha = nn.Sequential(
        nn.Linear(self.cf.dim_node_feature, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, self.cf.num_mix_component))


  def _break_graph(self, graph):
      # TODO implement GPU version
      '''
      break one graph into pieces, so the model only needs to learn step-wise generation
      In:
        graph: sparse tensor
          shape: num_nodes x num_nodes x dim_edge_feature
      '''
      num_nodes = graph.shape[0]
      Adj = graph.to_dense().squeeze(-1).ceil() #NOTE temporary fix
      Adj_l = torch.tril(Adj, diagonal=-1) # lower triangular part
      Adj_u = torch.tril(Adj.transpose(0,1), diagonal=-1) # upper triangular part
      adj_full = Adj.cpu().numpy()
      # make vectors for node features
      self.A_pad = torch.cat((F.pad(Adj_l,(0,self.cf.max_num_nodes-num_nodes),'constant',value=0.), 
                              F.pad(Adj_u,(0,self.cf.max_num_nodes-num_nodes),'constant',value=0.)
                              ), dim=-1)
      edges = []
      node_idx_gnn = []
      node_idx_feat = []
      label_ij = []
      label_ji = []
      subgraph_size = []
      subgraph_idx = []
      att_idx = []
      subgraph_count = 0
      start_idx_list = range(0, num_nodes, self.cf.stride)
      if self.cf.sample_subgraphs > 0: 
        start_idx_list = np.random.choice(start_idx_list, self.cf.sample_subgraphs, replace=True)
      for idx, start_idx in enumerate(start_idx_list):
        ### for each size-(start_idx+K) subgraph, we generate edges for the new block of K nodes
        if start_idx + self.cf.num_new_nodes > num_nodes: break
        ### get edges for GNN propagation
        adj_block = np.pad(adj_full[:start_idx, :start_idx], 
                           ((0, self.cf.num_new_nodes), (0, self.cf.num_new_nodes)),
                           'constant',
                           constant_values=1.0)  # assuming fully connected for the new block
        adj_block = torch.from_numpy(adj_block).to_sparse()
        edges += [adj_block.coalesce().indices().long()]
        ### attention index, existing nodes: 0, new nodes: 1, ..., K
        if start_idx == 0:
          att_idx += [np.arange(1, self.cf.num_new_nodes + 1).astype(np.uint8)]
        else:
          att_idx += [np.concatenate([np.zeros(start_idx).astype(np.uint8),
                                      np.arange(1, self.cf.num_new_nodes + 1).astype(np.uint8)])]
        ### get node feature index for GNN input
        # use inf to indicate the new nodes where input feature is zero
        if start_idx == 0:
          node_idx_feat += [np.ones(self.cf.num_new_nodes) * -1]
        else:
          node_idx_feat += [np.concatenate([np.arange(start_idx),
                            np.ones(self.cf.num_new_nodes) * -1])]
        ### get node index for GNN output
        idx_row_gnn, idx_col_gnn = np.meshgrid(np.arange(start_idx, start_idx + self.cf.num_new_nodes), 
                                               np.arange(start_idx + self.cf.num_new_nodes))
        idx_row_gnn = idx_row_gnn.reshape(-1, 1)
        idx_col_gnn = idx_col_gnn.reshape(-1, 1)
        node_idx_gnn += [np.concatenate([idx_row_gnn, idx_col_gnn],
                                        axis=1).astype(np.int64)]
        ### get predict label for i to j
        l_ij = adj_full[idx_row_gnn, idx_col_gnn] > 0
        label_ij += [l_ij.flatten().astype(np.uint8)]
        ### get predict label for j to i
        l_ji = adj_full[idx_col_gnn, idx_row_gnn] > 0
        label_ji += [l_ji.flatten().astype(np.uint8)]
        ### for loss
        subgraph_size += [start_idx + self.cf.num_new_nodes]
        subgraph_idx += [np.ones_like(label_ij[-1]).astype(np.int64) * subgraph_count]
        subgraph_count += 1
      ### adjust index base for subgraphs
      cum_size = np.cumsum([0] + subgraph_size).astype(np.int64)
      for i in range(len(edges)):
        edges[i] = edges[i] + cum_size[i]
        node_idx_gnn[i] = node_idx_gnn[i] + cum_size[i]
      self.edges = torch.cat(edges, dim=1).t().long()
      self.edges = self.edges.to(0)
      self.att_idx = torch.from_numpy(np.concatenate(att_idx)).long().to(0)
      self.node_idx_nodes = torch.from_numpy(np.concatenate(node_idx_feat)).long().to(0)
      self.new_nodes_idx = torch.from_numpy(np.concatenate(node_idx_gnn)).long().to(0)
      self.label_ij = torch.from_numpy(np.concatenate(label_ij)).float().to(0)
      self.label_ji = torch.from_numpy(np.concatenate(label_ji)).float().to(0)
      self.subgraph_idx = torch.from_numpy(np.concatenate(subgraph_idx)).long().to(0)
      self.subgraph_count = subgraph_count
      self.subgraph_idx_base = torch.tensor([0, subgraph_count], device=0).long()
      

  def _forward_training(self, graphs): 
    '''
    '''
    self._break_graph(graphs[0])
    #--- GNN for new nodes embedding 
    # all new nodes will have zero as node feature, indexed by -1
    node_features = F.pad(self.adj_vecs_to_nodes_state(self.A_pad), (0, 0, 0, 1), 'constant', value=0.0) 
    # create one-hot edge feature, each with size of (num_new_nodes+1)*2
    # [1 0 ...] if source node is not new, [0 1 0 ...] if source node is new node 1. [0 0 1 ...] etc
    att_idx = self.att_idx.view(-1, 1)
    att_edge_feature = torch.zeros(self.edges.shape[0], 2*self.cf.dim_att_edge).to(0)
    att_edge_feature = att_edge_feature.scatter(1, att_idx[[self.edges[:, 0]]], 1)
    att_edge_feature = att_edge_feature.scatter(1, att_idx[[self.edges[:, 1]]] + self.cf.dim_att_edge, 1)
    # get node states for new nodes !!! node_feature is shared among all subgraphs
    node_state = self.gne(node_features[self.node_idx_nodes], self.edges, att_edge_feature)
    #--- Pairwise prediction for the existance of the new edges
    diff = node_state[self.new_nodes_idx[:, 0], :] - node_state[self.new_nodes_idx[:, 1], :]
    # num_new_edges x 2 x num_mix_component
    log_theta = self.output_theta(diff).view(-1, 2, self.cf.num_mix_component)
    # num_new_edges x num_mix_component
    log_alpha = self.output_alpha(diff).view(-1, self.cf.num_mix_component)  
    return log_theta, log_alpha


  def _loss(self, log_theta, log_alpha, adj_loss_func,
            sum_order_log_prob=False, return_neg_log_prob=False, reduction="mean"):
    """
      Compute likelihood for mixture of Bernoulli model
      In:
        label: E X 1, see comments above
        log_theta: E X D, see comments above
        log_alpha: E X D, see comments above
        adj_loss_func: BinaryCrossEntropy loss
        subgraph_idx: E X 1, see comments above
        subgraph_idx_base: B+1, cumulative # of edges in the subgraphs associated with each batch
        num_canonical_order: int, number of node orderings considered
        sum_order_log_prob: boolean, if True sum the log prob of orderings instead of taking logsumexp 
          i.e. log p(G, pi_1) + log p(G, pi_2) instead of log [p(G, pi_1) + p(G, pi_2)]
          This is equivalent to the original GRAN loss.
        return_neg_log_prob: boolean, if True also return neg log prob
        reduction: string, type of reduction on batch dimension ("mean", "sum", "none")
      Returns:
        loss (and potentially neg log prob)
    """
    self.device = self.label_ij.device
    num_graphs = self.subgraph_idx_base.shape[0] - 1
    num_subgraph = self.subgraph_idx_base[-1] # == subgraph_idx.max() + 1
    num_edges = log_theta.shape[0]
    adj_ij_loss = torch.stack([adj_loss_func(log_theta[:, 0, i], self.label_ij) 
                            for i in range(self.cf.num_mix_component)], dim=1)
    adj_ji_loss = torch.stack([adj_loss_func(log_theta[:, 1, i], self.label_ji) 
                            for i in range(self.cf.num_mix_component)], dim=1)
    adj_loss = adj_ij_loss+adj_ji_loss
    # num of new edges for each subgraph
    const = torch.zeros(num_subgraph).to(self.device) # S
    const = const.scatter_add(0, self.subgraph_idx,
                              torch.ones_like(self.subgraph_idx).float())
    # sum over all edges in each subgraph
    reduce_adj_loss = torch.zeros(num_subgraph, self.cf.num_mix_component).to(self.device)
    reduce_adj_loss = reduce_adj_loss.scatter_add(
        0, self.subgraph_idx.unsqueeze(1).expand(-1, self.cf.num_mix_component), adj_loss)
    # average over all edges in the same subgraph, then softmax
    reduce_log_alpha = torch.zeros(num_subgraph, self.cf.num_mix_component).to(self.device)
    reduce_log_alpha = reduce_log_alpha.scatter_add(
        0, self.subgraph_idx.unsqueeze(1).expand(-1, self.cf.num_mix_component), log_alpha)
    reduce_log_alpha = reduce_log_alpha / const.view(-1, 1)
    reduce_log_alpha = F.log_softmax(reduce_log_alpha, -1) # log probability
    log_prob = -reduce_adj_loss + reduce_log_alpha
    log_prob = torch.logsumexp(log_prob, dim=1) # num_subgraphs 
    bc_log_prob = torch.zeros([num_graphs]).to(self.device) # num_graphs
    bc_idx = torch.arange(num_graphs).to(self.device) # num_graphs
    bc_const = torch.zeros(num_graphs).to(self.device)
    bc_size = (self.subgraph_idx_base[1:] - self.subgraph_idx_base[:-1]) # num_graphs
    bc_size = torch.repeat_interleave(bc_size, 1) # num_graphs
    bc_idx = torch.repeat_interleave(bc_idx, bc_size) # S
    bc_log_prob = bc_log_prob.scatter_add(0, bc_idx, log_prob)
    # loss must be normalized for numerical stability
    bc_const = bc_const.scatter_add(0, bc_idx, const)
    bc_loss = (bc_log_prob / bc_const)
    bc_log_prob = bc_log_prob.reshape(num_graphs,1)
    bc_loss = bc_loss.reshape(num_graphs,1)
    if sum_order_log_prob:
      b_log_prob = torch.sum(bc_log_prob, dim=1)
      b_loss = torch.sum(bc_loss, dim=1)
    else:
      b_log_prob = torch.logsumexp(bc_log_prob, dim=1)
      b_loss = torch.logsumexp(bc_loss, dim=1)
    # probability calculation was for lower-triangular edges
    # must be squared to get probability for entire graph
    b_neg_log_prob = -2*b_log_prob
    b_loss = -b_loss
    if reduction == "mean":
      neg_log_prob = b_neg_log_prob.mean()
      loss = b_loss.mean()
    elif reduction == "sum":
      neg_log_prob = b_neg_log_prob.sum()
      loss = b_loss.sum()
    else:
      assert reduction == "none"
      neg_log_prob = b_neg_log_prob
      loss = b_loss
    if return_neg_log_prob:
      return loss, neg_log_prob
    else:
      return loss

 
  def _forward_sampling(self, gene_codes, graphs_size):
    """
    The gradient will explode, if training with this 
    NOTE!!! if num_new_nodes >= 2, there is an overlap in updating !!! 
    In:
      gene_codes: tensor
        shape: num_graphs x dim_gene_codes
      graphs_size: list
        shape: num_graphs x 1
    Out:
      existance_score: tensor 
        shape: num_graphs x N_max x N_max x 1
      features: tensor
        shape: num_graphs x N_max x N_max x dim_edge_feature
    """
    d = 0#gene_codes.device
    num_graphs = len(graphs_size)
    with torch.no_grad():
      ### cache node state for speed up
      A = torch.zeros(num_graphs, self.cf.max_num_nodes, self.cf.max_num_nodes).to(d)
      node_state = torch.zeros(num_graphs, self.cf.max_num_nodes, self.cf.dim_node_feature).to(d)
      for start_idx in range(0, self.cf.max_num_nodes, self.cf.stride):
        end_idx = start_idx + self.cf.num_new_nodes
        if end_idx > (self.cf.max_num_nodes-1): break
        #--- input for GNN, num_graphs x end_idx x dim_node_feature  &&  num_graphs x end_idx x end_idx
        node_state_in = F.pad(node_state[:, :start_idx, :], (0, 0, 0, self.cf.num_new_nodes), 'constant', value=.0)
        adj = F.pad(A[:, :start_idx, :start_idx], (0, self.cf.num_new_nodes, 0, self.cf.num_new_nodes), 'constant', value=1.0)  
        edges = torch.cat([ adj[i].to_sparse().coalesce().indices() + i * end_idx
                              for i in range(num_graphs) ], dim=1).t()
        #- create one-hot feature for all edges
        # end_idx -> num_graphs*end_idx x 1, dim_att_edge has to be at least the size of num_new_nodes + 1
        # [0 1] if source node is new node 1, [0 0 1] if source node is new node 2, otherwise [1 0]
        # similar to above, but for target nodes
        att_idx = torch.cat([torch.zeros(start_idx).long(), 
                             torch.arange(1, self.cf.num_new_nodes + 1)]).to(d)
        att_idx = att_idx.view(1, -1).expand(num_graphs, -1).contiguous().view(-1, 1) 
        att_edge_feat = torch.zeros(edges.shape[0], 2*self.cf.dim_att_edge).to(d) # num_edges x dim_edge_feature  
        att_edge_feat = att_edge_feat.scatter(1, att_idx[[edges[:, 0]]], 1) # source, num_edges x 1
        att_edge_feat = att_edge_feat.scatter(1, att_idx[[edges[:, 1]]] + self.cf.dim_att_edge, 1)
        #--- GNN propagation
        node_state_out = self.gne(node_state_in.view(-1, self.cf.dim_node_feature), edges, att_edge_feat)
        node_state_out = node_state_out.view(num_graphs, end_idx, -1)
        # new potential edges
        idx_row, idx_col = torch.meshgrid(torch.arange(start_idx, end_idx), torch.arange(end_idx))
        diff = node_state_out[:,idx_row, :] - node_state_out[:,idx_col, :]
        # num_graphs x num_new_nodes x end_idx x dim_node_feature -> ... x dim_node_feature
        ## -> ... x 2*num_mixture   &&   .. x new_num_nodes*num_new_nodes x num_mixture
        diff = diff.view(-1, node_state_out.shape[2])
        log_theta = self.output_theta(diff).view(num_graphs, self.cf.num_new_nodes, end_idx, 2, self.cf.num_mix_component)
        log_alpha = self.output_alpha(diff).view(num_graphs, -1, self.cf.num_mix_component)
        prob_alpha = F.softmax(log_alpha.mean(dim=1), -1) # mixture components sum to 1
        #--- original sampling scheme, first mixture component, then bernulli
        alpha = torch.multinomial(prob_alpha, 1).squeeze(dim=1).long()
        prob_ij, prob_ji = [], []
        for bb in range(num_graphs):
          prob_ij += [torch.sigmoid(log_theta[bb, :, :, 0, alpha[bb]])]
          prob_ji += [torch.sigmoid(log_theta[bb, :, :, 1, alpha[bb]])]
        prob_ij, prob_ji = torch.stack(prob_ij, dim=0), torch.stack(prob_ji, dim=0)
        prob_edges = torch.stack([prob_ij, prob_ji], dim=-1)
        #--- sampling
        edges_ij, edges_ji = torch.bernoulli(prob_edges[:,:,:,0]), torch.bernoulli(prob_edges[:,:,:,1])
        #--- updating
        A[:, start_idx:end_idx, :end_idx] = edges_ij
        A[:, :end_idx, start_idx: end_idx] = edges_ji.transpose(1,2)
        node_vecs = torch.cat([A[:, start_idx: end_idx, :], A[:, :, start_idx: end_idx].transpose(1,2)], dim=-1)
        node_state[:, start_idx: end_idx, :] = self.adj_vecs_to_nodes_state(node_vecs)
      return A, None 








# NOTE it's not working :(
import numpy as np
class GraphGenRnn(nn.Module):
  '''
  Graph generator based on recurrent neural network
  Params:
  '''
  def __init__(self,dim_gene_code,
                    dim_edge_feature,
                    dim_rnn_node,
                    dim_rnn_edge,
                    num_layer):
    super(GraphGenRnn, self).__init__()
    self.dim_rnn_node = dim_rnn_node
    self.dim_rnn_edge = dim_rnn_edge
    self.dim_edge_feature = dim_edge_feature
    self.num_layer = num_layer
    # node-level
    self.node_entry_func = nn.Sequential(*[nn.Linear(dim_gene_code, dim_rnn_node), nn.ReLU()])
    self.node_rnn = nn.LSTM(input_size=1, hidden_size=dim_rnn_node, num_layers=num_layer)
    self.node_exit_func = nn.Sequential(*[nn.Linear(dim_rnn_node, 32), nn.ReLU()])
    # edge-level
    self.edge_entry_func = nn.Sequential(*[nn.Linear(32, dim_rnn_edge), nn.ReLU()])
    self.edge_rnn = nn.LSTM(input_size=1, hidden_size=dim_rnn_edge, num_layers=num_layer)
    self.edge_exit_func = nn.Sequential(*[nn.Linear(dim_rnn_edge, 16), nn.ReLU()])
    # out-put
    self.edge_existance_score = nn.Sequential(*[nn.Linear(16, 8), nn.ReLU(),
                                                nn.Linear(8, 2), nn.Sigmoid()])
    self.edge_feature = nn.Sequential(*[nn.Linear(16, 8), nn.ReLU(),
                                        nn.Linear(8, 2*dim_edge_feature)])


  def forward(self, gene_codes, graphs_size):
    """
    generate a batch of graphs 
    In:
      gene_codes: tensor
        shape: num_graphs x dim_gene_codes
      graphs_size: list
        shape: num_graphs x 1
    Out:
      existance_score: sparse tensor 
        shape: num_graphs x N_max x N_max x 1
      features: sparse tensor
        shape: num_graphs x N_max x N_max x dim_edge_feature
    """
    num_graphs = gene_codes.shape[0]
    graphs_size_max  = max(graphs_size)
    num_nodes_max = max(graphs_size)+1
    #print(num_graphs)
    #print(graphs_size)
    #print(len(graphs_size))
    #print(max(graphs_size))
    #print('###################\n\n')

    #------- node-level
    node_entry_point = self.node_entry_func(gene_codes) # num_graphs x dim_rnn_node
    # initial node rnn states, both of size:  num_layers x num_graphs x dim_rnn_node
    h0_node = node_entry_point.unsqueeze(0).repeat(self.num_layer,1,1)
    c0_node = torch.zeros_like(h0_node)
    # flip the size, becuase rnn sequence packing requires descending order...
    graphs_idx = np.flip(np.argsort(graphs_size))
    descend_order_graphs_size = [graphs_size[i] for i in list(graphs_idx)]
    # constant input, list of 0s with length of graph's size, as:
    #
    #  [[0, 0, 0, ...]    size: graphs_idx[0] 
    #   [0, 0, ...]
    #   ...
    #  ] 
    #         num_graphs x sequences_length x dim_input
    node_rnn_in = [torch.zeros((l,1), dtype=gene_codes.dtype, device=gene_codes.device)
                        for l in descend_order_graphs_size]
    node_rnn_in = nn.utils.rnn.pack_sequence(node_rnn_in)
    node_rnn_output, (hn_node, cn_node) = self.node_rnn(node_rnn_in,
                                                        (h0_node, c0_node))
    # this specifies which graph this node belongs to
    # size is the same as the output from node rnn : sum(graphs_size) x dim_rnn_edge
    node_graph_idx = [torch.tensor([graphs_idx[i]]*l) for i, l in enumerate(descend_order_graphs_size)]
    node_graph_idx = nn.utils.rnn.pack_sequence(node_graph_idx)
    node_graph_idx = node_graph_idx.data

    # flip everthing, because rnn utils function pack_sequence requires descending ordering ...
    node_graph_idx = torch.flip(node_graph_idx,[0])
    node_rnn_output_data = torch.flip(node_rnn_output.data,[0])
    node_rnn_output_batch_sizes = torch.flip(node_rnn_output.batch_sizes,[0])

    #print(node_graph_idx)
    #print(node_rnn_output_data.shape)
    #print(node_rnn_output_batch_sizes)

    #------- edge-level
    edge_entry_point = self.edge_entry_func(self.node_exit_func(node_rnn_output_data)) 
    # initial edge rnn states, both of size: sum(graphs_size) x dim_rnn_edge
    h0_edge = edge_entry_point.unsqueeze(0).repeat(self.num_layer,1,1)
    c0_edge = torch.zeros_like(h0_edge)
    # node_rnn_output_batch_sizes specifies the number of iterations/edges to generate for this node
    #edges_size = torch.range(graphs_size_max,1,-1).repeat_interleave(node_rnn_output_batch_sizes)
    edges_size = np.repeat(np.arange(graphs_size_max, 0, -1), node_rnn_output_batch_sizes)
    #print(edges_size)
    #print(edges_size.shape)

    # constant input, list of 0s with length of edges size 
    edge_rnn_in = [torch.zeros((int(length),1), dtype=gene_codes.dtype, device=gene_codes.device)
                        for length in edges_size.tolist()] 
    edge_rnn_in = nn.utils.rnn.pack_sequence(edge_rnn_in)
    edge_rnn_output, (hn_edge, cn_edge) = self.edge_rnn(edge_rnn_in, (h0_edge, c0_edge))
    edge_exit_point = self.edge_exit_func(edge_rnn_output.data)
    edge_existance_score = self.edge_existance_score(edge_rnn_output.data)
    edge_feature = self.edge_feature(edge_rnn_output.data).view(-1, 2, self.dim_edge_feature)

    # this specifies which graph this edge belongs to
    edge_graph_idx = [torch.tensor([node_graph_idx[i]]*l) for i, l in enumerate(edges_size)]
    edge_graph_idx = nn.utils.rnn.pack_sequence(edge_graph_idx)
    edge_graph_idx = edge_graph_idx.data
    #print(edge_graph_idx)
    #print(edge_graph_idx.shape)

    # this specifies the row node idx, which is the same as the edge size
    edge_row_idx = [torch.tensor([edges_size[i]]*l) for i, l in enumerate(edges_size)]
    edge_row_idx = nn.utils.rnn.pack_sequence(edge_row_idx)
    edge_row_idx = edge_row_idx.data
    #print(edge_row_idx)
    #print(edge_row_idx.shape)
    
    # this specifies the row node idx, which is the same as the edge size
    edge_column_idx = [torch.arange(0,l) for l in edges_size]
    edge_column_idx = nn.utils.rnn.pack_sequence(edge_column_idx)
    edge_column_idx = edge_column_idx.data
    #print(edge_column_idx)
    #print(edge_column_idx.shape)
    
    indices_lower = torch.stack([edge_graph_idx, edge_row_idx, edge_column_idx])
    indices_upper = torch.stack([edge_graph_idx, edge_column_idx, edge_row_idx])
    indices = torch.cat([indices_lower, indices_upper], dim=-1).to(0)
    #print(indices_lower)
    #print(indices_lower.shape)
    #print(indices.shape)
    values_existance = torch.cat([edge_existance_score[:,0], edge_existance_score[:,1]],dim=0)
    values_feature = torch.cat([edge_feature[:,0,:], edge_feature[:,1,:]],dim=0)
     
    # put layers of edges into matrix
    existance_score = torch.sparse_coo_tensor(indices, values_existance,
                                              (num_graphs, num_nodes_max, num_nodes_max))
    features = torch.sparse_coo_tensor(indices, values_feature, 
                                       (num_graphs, num_nodes_max, num_nodes_max, self.dim_edge_feature))
    #print(existance_score.shape)
    #print(features.shape)
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




 














