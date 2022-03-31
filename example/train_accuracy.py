import torch
import dgl
import fastgraph

def load_accuracy_data(dataset_name, root_path):
    dataset = fastgraph.dataset(dataset_name, root_path)
    graph = dataset.to_dgl_graph()
    valid_set = dataset.valid_set
    test_set = dataset.test_set
    return (graph, valid_set, test_set, dataset.feat, dataset.label)

class Accuracy:
    def __init__(self, graph, valid_set, test_set, feat, label, fanout, batch_size, sample_device):
        # XXX: function "to" is
        #        not useful for samgraph while sample device is different GPU with training
        self.graph = graph #.to(sample_device)
        self.valid_set = valid_set #.to(sample_device)
        self.test_set = test_set #.to(sample_device)
        self.feat = feat
        self.label = label
        self.fanout = fanout
        self.batch_size = batch_size
        self.sample_device = sample_device
        self.sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
        # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(fanout))

    def __load_subtensor(self, input_nodes, output_nodes, train_device):
        input_nodes = input_nodes.to(self.feat.device)
        output_nodes = output_nodes.to(self.label.device)
        batch_inputs = torch.index_select(
            self.feat, 0, input_nodes.long()).to(train_device)
        batch_labels = torch.index_select(
            self.label, 0, output_nodes.long()).to(train_device)
        return batch_inputs, batch_labels

    # References:
    #     From DGL source code: examples/pytorch/ogb/ogbn-products/graphsage/main.py
    def __evaluate(self, model, eval_nids, train_device):
        total = 0
        total_correct = 0
        model.eval()
        with torch.no_grad():
            dataloader = dgl.dataloading.NodeDataLoader(
                self.graph,
                eval_nids,
                self.sampler,
                device=self.sample_device,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False
            )
            for input_nodes, output_nodes, blocks in dataloader:
                blocks = [block.int().to(train_device) for block in blocks]
                batch_inputs, batch_labels = self.__load_subtensor(input_nodes, output_nodes,
                                                      train_device)
                total += len(batch_labels)
                outputs = model(blocks, batch_inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == batch_labels.data).sum().item()
        model.train()
        acc = 1.0 * total_correct / total
        return acc

    def valid_acc(self, model, train_device):
        return self.__evaluate(model, self.valid_set, train_device)

    def test_acc(self, model, train_device):
        return self.__evaluate(model, self.test_set, train_device)

