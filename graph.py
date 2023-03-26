
class Graph:

    def __init__(self):
        self.nodes = []
        self.num_nodes = 0
        return

    def add_node(self, name, pointers_to={}, pointers_from={}):
        if self.node_in_graph(name):
            return False

        new_node = GraphNode(name)
        self.nodes.append(new_node)

        for node_name in pointers_to:
            node = self.get_node(node_name)
            weight = pointers_to[node_name]
            node.add_pointer(new_node, weight)

        for node_name in pointers_from:
            node = self.get_node(node_name)
            weight = pointers_from[node_name]
            new_node.add_pointer(node, weight)

        self.num_nodes += 1
        return True

    def get_node(self, name):
        for node in self.nodes:
            if name == node.name:
                return node
        return None

    def node_in_graph(self, name):
        node = self.get_node(name)
        return node is not None


class GraphNode:

    def __init__(self, name):
        self.name = name
        self.pointers = {}
        return

    def add_pointer(self, node, weight):
        self.pointers[node] = weight
        return
