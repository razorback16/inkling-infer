"""Check captured MLX parameter identities without rebuilding dotted weight paths."""
from operator import is_, itemgetter


class ParameterTree:
    def __init__(self, root):
        self.records = []
        self._record(root)

    def _record(self, node):
        if isinstance(node, dict):
            keys = tuple(node)
            children = tuple(node.values())
            getter = itemgetter(*keys) if len(keys) > 1 else None
            self.records.append((node, keys, children, getter))
        elif isinstance(node, list):
            children = tuple(node)
            self.records.append((node, None, children, None))
        elif isinstance(node, tuple):
            children = node
        else:
            return
        for child in children:
            self._record(child)

    def matches(self):
        for node, keys, children, getter in self.records:
            if len(node) != len(children):
                return False
            if keys is None:
                if not all(map(is_, node, children)):
                    return False
            elif len(keys) == 1:
                if node.get(keys[0]) is not children[0] or keys[0] not in node:
                    return False
            elif getter is not None:
                try:
                    values = getter(node)
                except KeyError:
                    return False
                if not all(map(is_, values, children)):
                    return False
        return True
