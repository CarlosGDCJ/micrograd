from graphviz import Digraph


class Value:
    """Differentiable float"""

    def __init__(self, value, parents=(), op=""):
        self.data = value
        self.parents = parents
        self.op = op
        self.grad = 0
        # Backward is a function because it only starts after the forward propagation
        # We can't compute during it, it's a callback

        # We use the current gradient to set the gradients of the parents
        # At the very end of the computational graph, the gradient of the last node will be 1
        # That's the base case
        # From then we see which operation originated the last node

        # --> l is the last (dl/dl = 1)
        # e.g.: a + b = c
        #       c * d = l

        #       Contribution of c and d
        #       dl/dc = dl/dl * dl/dc = 1 * d = d
        #       dl/dd = dl/dl * dl/dd = 1 * c = c

        #       Contribution of a and b
        #       dl/da = dl/dc * dc/da = d * 1 = d
        #       dl/db = dl/dc * dc/db = d * 1 = d

        # We used the gradient of l to set the gradients of its parents (d and c)
        # We used the gradient of c to set the gradients of its parents (a and b)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value: {self.data}"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, parents=(self, other), op="+")

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        # if we don't do it like this (self - other) we get a recrusion error
        # cause we'd call this function infinetly
        # -other calls other.__neg__()
        return self + (-other)

    def __rsub__(self, other):
        return self - other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, parents=(self, other), op="*")

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        # can't do the standard self / other on r methods because unlike
        # multiplication, division is not commutative
        return (self**-1) * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Exponent not supported"
        return Value(self.data**other, parents=(self,), op="**")

    def __neg__(self):
        return -1 * self

    def _make_graph(self):
        nodes, edges = set(), set()

        def explore(node):
            if node not in nodes:
                nodes.add(node)
                for parent in node.parents:
                    edges.add((parent, node))
                    explore(parent)

        explore(self)
        return nodes, edges

    # https://github.com/karpathy/micrograd/blob/master/trace_graph.ipynb
    def draw_graph(self):
        nodes, edges = self._make_graph()

        dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})

        for n in nodes:
            # create a value node
            dot.node(
                name=str(id(n)),
                label=f"{{data: {n.data:.4f} | grad: {n.grad:.4f}}}",
                shape="record",
            )
            if n.op:
                # create an operation node and point the value node to it
                dot.node(name=str(id(n)) + n.op, label=n.op)
                dot.edge(str(id(n)) + n.op, str(id(n)))

        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2.op)

        return dot
