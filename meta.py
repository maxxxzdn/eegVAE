import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean

class EdgeModel(torch.nn.Module):
    def __init__(self, dims):
        super(EdgeModel, self).__init__()
        self.edge_mlp = Seq(Lin(dims["n_f"]*2 + dims["e_f"] + dims["u_f"],  dims["hidden"]), ReLU(), Lin(dims["hidden"], dims["e_f"]))

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)

class NodeModel(torch.nn.Module):
    def __init__(self, dims):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = Seq(Lin(dims["n_f"] + dims["e_f"], dims["hidden"]), ReLU(), Lin(dims["hidden"], dims["hidden"]))
        self.node_mlp_2 = Seq(Lin(dims["n_f"] + dims["hidden"] + dims["u_f"], dims["hidden"]), ReLU(), Lin(dims["hidden"], dims["n_f"]))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u[batch]], dim=1)
        return self.node_mlp_2(out)

class GlobalModel(torch.nn.Module):
    def __init__(self, dims):
        super(GlobalModel, self).__init__()
        self.global_mlp = Seq(Lin(dims["u_f"]+dims["n_f"], dims["hidden"]), ReLU(), Lin(dims["hidden"], dims["u_f"]))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        return self.global_mlp(out)