import dgl
import samgraph.torch as sam


def to_dgl_graphs(batch_key, num_layers):
    feat = sam.get_graph_feat(batch_key)
    label = sam.get_graph_label(batch_key)
    graphs = []
    for i in range(num_layers):
        row = sam.get_graph_row(batch_key, i)
        col = sam.get_graph_col(batch_key, i)

        graphs.append(dgl.graph((row, col)))

    return graphs, feat, label
