class Value:
    """Differentiable float"""

    def __init__(self, value, parents=(), op=""):
        self.data = value
        self.parents = parents
        self.op = op

    def __repr__(self):
        if self.parents:
            return (
                f"Value: {self.data} Created by {self.op} applied to"
                f" {[x.data for x in self.parents]}"
            )

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

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Exponent not supported"
        return Value(self.data**other, parents=(self,), op="**")

    def __neg__(self):
        return -1 * self
