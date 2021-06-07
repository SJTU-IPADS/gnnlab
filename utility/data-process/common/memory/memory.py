TB = 1 * 1024 * 1024 * 1024 * 1024
GB = 1 * 1024 * 1024 * 1024
MB = 1 * 1024 * 1024
KB = 1 * 1024

BILLION = 1 * 10**9
MILLION = 1 * 10**6
THOUSAND = 1 * 10**3
INT_BYTES = 4
FLOAT_BYTES = 4


def sz_format(sz):
    if sz >= TB:
        return "{:<6.2f} TB".format(sz / TB)
    elif sz >= GB:
        return "{:<6.2f} GB".format(sz / GB)
    elif sz >= MB:
        return "{:<6.2f} MB".format(sz / MB)
    elif sz >= KB:
        return "{:<6.2f} KB".format(sz / KB)
    else:
        return "{:<6.2f} B".format(sz)


class Graph:
    def __init__(self, name, num_nodes, num_edges, feat_dim):
        self.name = name
        self.num_nodes = num_nodes * 1.0
        self.num_edges = num_edges * 1.0
        self.feat_dim = feat_dim * 1.0

    def __str__(self):
        graph_sz = sz_format((self.num_nodes + self.num_edges) * INT_BYTES)
        feat_sz = sz_format((self.num_nodes * self.feat_dim) * FLOAT_BYTES)

        return "{:15s} | topology: {:s} | feat: {:s}".format(self.name, graph_sz, feat_sz)


Papers100M = Graph('Papers100M', 111059956, 1726745828, 128)
ComFriendster = Graph('com-friendster', 65608366, 1806067135, 256)
AlipayGraph = Graph('AlipayGraph', 4 * BILLION, 26 * BILLION, 128 * 4)

graph_list = [Papers100M, ComFriendster, AlipayGraph]


def run():
    for graph in graph_list:
        print(graph)


if __name__ == '__main__':
    run()
