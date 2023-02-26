import tensorflow as tf
import numpy as np

class graph(object):
    def __init__(self, node_num = 0, label = None, name = None):
        self.node_num = node_num
        self.label = label
        self.name = name
        self.features = []
        self.succs = []
        self.preds = []
        if (node_num > 0):
            for _ in range(node_num):
                self.features.append([])
                self.succs.append([])
                self.preds.append([])

    def add_node(self, feature = []):
        self.node_num += 1
        self.features.append(feature)
        self.succs.append([])
        self.preds.append([])
        
    def add_edge(self, u, v):
        self.succs[u].append(v)
        self.preds[v].append(u)

class Model(object):
    def __init__(self, model_path, load_type="pb"):
        self.sess = None
        if load_type == "pb": self._loadModelFromPb(model_path)
        # undo load for others model format
        assert self.sess != None

        self.graph_input = self.sess.graph.get_tensor_by_name("import/graph_input:0")
        self.graph_mask_input = self.sess.graph.get_tensor_by_name("import/graph_mask_input:0")
        self.graph_embedding = self.sess.graph.get_tensor_by_name("import/graph_embedding:0")

    def _loadModelFromPb(self, model_path):
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            with open(model_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def)

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            self.sess = sess

    def _parseInputFormatForGNN(self, g):
        basic_block_num = g.node_num
        feature_dim = len(g.features[0])
        X1_input = np.zeros((1, basic_block_num, feature_dim))
        node1_mask = np.zeros((1, basic_block_num, basic_block_num))
        for u in range(basic_block_num):
            X1_input[0, u, :] = np.array(g.features[u])
            for v in g.succs[u]:
                node1_mask[0, u, v] = 1
        return X1_input, node1_mask

    def predict(self, g):
        graph_input, graph_mask_input = self._parseInputFormatForGNN(g)
        output_embedding = self.sess.run(self.graph_embedding, feed_dict = {
            self.graph_input: graph_input,
            self.graph_mask_input: graph_mask_input
        })
        return output_embedding
