"""Validate compiled MLX module and array identities in one object traversal."""
import gc
import operator

import mlx.core as mx
from mlx.utils import tree_flatten


class CapturedParameters:
    def capture(self, model):
        nodes = []
        stack = [model]
        seen = set()
        while stack:
            node = stack.pop()
            if id(node) in seen or not isinstance(node, (dict, list, tuple)):
                continue
            seen.add(id(node))
            nodes.append(node)
            stack.extend(node.values() if isinstance(node, dict) else node)
        self.nodes = tuple(nodes)
        # CPython exposes these containers' direct references in C. Verify that
        # the runtime reported every array before using that path for cache reuse.
        current = gc.get_referents(*self.nodes)
        tracked = {id(a) for a in current}
        arrays = self._arrays(model)
        self.direct = all(id(a) in tracked for a in arrays)
        self.values = current if self.direct else arrays

    @staticmethod
    def _arrays(model):
        return [v for _,v in tree_flatten(model) if isinstance(v,mx.array)]

    def matches(self, model):
        current = gc.get_referents(*self.nodes) if self.direct else self._arrays(model)
        return len(current) == len(self.values) and all(map(operator.is_,current,self.values))
