import torch
import torch as th
from torch import nn
from torch.nn import init
import torch.nn.functional as F

import dgl
from dgl import function as fn
from dgl.nn.pytorch.softmax import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.base import DGLError


from config import HIDDEN_DIM
from config import EDGE_TYPE as ETYPE

# pylint: disable=W0235
class FSRGraphConv(nn.Module):
    r"""Friend Story Ranking Graph Conv that takes edge features into consideration
    Note that in_feats == node_ftr_dim + edge_ftr_dim
    """
    def __init__(self,
                 in_feats,
                 edge_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None):
        super().__init__()
        if norm not in ('none', 'both', 'right'):
            raise DGLError('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._edge_feats = edge_feats
        self._out_feats = out_feats
        self._norm = norm

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats + edge_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

        self.W = nn.Linear(in_feats + out_feats, out_feats)

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def message_func(self, edges):
        # get edge attr and node attr
        feat = th.cat([edges.src['h'], edges.data['eftr']], dim = -1)
        return {'m': feat}

    def reduce_func(self, nodes):
        feat = nodes.mailbox['m']
        feat = th.mean(feat, dim=1)
        h = th.matmul(feat, self.weight)  # h contains info from both nbhd and edge
        return {'h_neigh': h}


    def forward(self, block, h):
        with block.local_scope():
            h_src = h
            h_dst = h[:block.number_of_dst_nodes()]
            block.srcdata['h'] = h_src
            block.dstdata['h'] = h_dst



            block.update_all(self.message_func, self.reduce_func)

            # cat self and nbhd avg
            h_prime = self.W(th.cat(
                [block.dstdata['h'], block.dstdata['h_neigh']], 1))


            if self.bias is not None:
                h_prime = h_prime + self.bias

            if self._activation is not None:
                h_prime = self._activation(h_prime)

        return h_prime


    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}, eftr={_edge_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)

class CTRPredictor(nn.Module):
    def __init__(self, num_classes, in_features, similarity = 'dot'):
        super().__init__()
        self.W = nn.Linear(2 * in_features, num_classes)
        self.apply_edges_dot = dgl.function.u_dot_v('norm_x', 'norm_x', 'score')
        self.similarity = similarity

    def apply_edges_mul(self, edges):
        data = th.cat([edges.src['x'], edges.dst['x']])
        return {'score': self.W(data)}

    def forward(self, edge_subgraph, x):
        apply_edges = self.apply_edges_mul if self.similarity == 'mul' else self.apply_edges_dot
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            edge_subgraph.ndata['norm_x'] = F.normalize(x, p=2, dim=-1)
            edge_subgraph.apply_edges(apply_edges)
            return edge_subgraph.edata['score']

class FSRGNN(nn.Module):
    def __init__(self, n_dim, e_dim, hidden_dim = HIDDEN_DIM, num_classes = 2, similarity = 'dot'):
        super().__init__()
        self.layer1 = FSRGraphConv(n_dim, e_dim, hidden_dim)
        self.layer2 = FSRGraphConv(hidden_dim, e_dim, hidden_dim)
        self.predictor = CTRPredictor(num_classes, hidden_dim, similarity)

    def forward(self, edge_subgraph, blocks, x):
        x = self.layer1(blocks[0], x)
        x = F.elu(x)
        x = self.layer2(blocks[1], x)
        pred = self.predictor(edge_subgraph, x)
        return pred

class LocalFSRGraphConv(FSRGraphConv):
    def __init__(self, in_feats, edge_feats, out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None):
        super().__init__(in_feats, edge_feats, out_feats, norm=norm, weight=weight, bias=bias, activation=activation)


    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.update_all(self.message_func, self.reduce_func)

            h_prime = self.W(th.cat(
                [g.dstdata['h'], g.dstdata['h_neigh']], 1))


            if self.bias is not None:
                h_prime = h_prime + self.bias

            if self._activation is not None:
                h_prime = self._activation(h_prime)

        return h_prime

class ELRGNN(nn.Module):
    def __init__(self, n_dim, e_dim, hidden_dim = HIDDEN_DIM):
        super().__init__()
        self.layer1 = LocalFSRGraphConv(n_dim, e_dim, hidden_dim)
        self.layer2 = LocalFSRGraphConv(hidden_dim, e_dim, hidden_dim)
        self.e_W = nn.Linear(2 * hidden_dim + e_dim, hidden_dim)
        self.fc = nn.Linear(4*hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + hidden_dim, 1)
        self.eftr_fc = nn.Linear(e_dim, hidden_dim)

    def get_edge_repr(self, edges):
        e_emb = th.cat([edges.src['nemb'], edges.dst['nemb'], edges.data['eftr']], dim=-1)
        return {'eemb': self.e_W(e_emb)}


    def forward(self, samples):
        preds = []
        for g, types in samples:
            type0, type1, type2, type3 = types
            x = g.ndata['nftr']
            x = self.layer1(g, x)
            x = F.tanh(x)
            x = self.layer2(g, x)
            g.ndata['nemb'] = x  # assign node embedding to 'nemb'
            g.apply_edges(self.get_edge_repr)  # get edge embedding
            eftr = g.edata['eftr'][type0, :]
            eemb = g.edata['eemb']
            e_uv = eemb[type0, :]
            e_uuf = eemb[type1, :]
            e_vfv = eemb[type2, :]
            e_ufvf = eemb[type3, :]
            attn_u = th.nn.Softmax(th.mul(e_uuf, e_uv.transpose())).reshape((e_uuf.shape[0], 1))
            attn_v = th.nn.Softmax(th.mul(e_vfv, e_uv.transpose())).reshape((e_vfv.shape[0], 1))
            fc_in = th.cat([e_uv, th.mean(e_uuf, dim = 0, keepdim=True), th.sum(attn_u * e_uuf, dim = 0, keepdim=True),
                th.sum(attn_v * e_vfv, dim = 0, keepdim=True), th.mean(e_ufvf, dim = 0, keepdim=True)],
                dim = -1)
            score = self.fc(fc_in)
            score = F.tanh(score)
            eftr_emb = self.eftr_fc(eftr)
            eftr_emb = F.tanh(eftr_emb)
            score = th.cat([score, eftr_emb], dim = -1)
            score = self.fc2(score)
            preds.append(score)
        return th.cat(preds, dim = -1).flatten()