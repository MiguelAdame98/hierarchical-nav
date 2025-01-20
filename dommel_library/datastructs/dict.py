class Dict(dict):
    """ Custom dictionary implementation
    Adds:
    dot.notation access to dictionary attributes
    update of nested dictionaries
    """

    def __getattr__(self, attr):
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError(attr)

    def __getitem__(self, key):
        try:
            val = dict.__getitem__(self, key)
            if type(val) is dict:
                val = Dict(val)
                self[key] = val
        except Exception:
            return None
        return val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)

    def update(self, *dicts, **other):
        """ Update the self dict
        :param dicts: optional list of dictionaries to include
        :param other: key value pairs to include
        The original dict is overwritten left to right, so the right most dict
        overwrites te left most, and key values overwrite that again.
        """
        others = {}
        [others.update(d) for d in dicts]
        others.update(other)
        for k, v in others.items():
            if isinstance(v, dict):
                if k not in self.keys():
                    self[k] = Dict(v)
                else:
                    self[k].update(v)
            else:
                self[k] = v
        return None

    def dict(self):
        """ Convert to regular dict (e.g. to export to file) """
        result = dict()
        for k, v in self.items():
            if type(v) is Dict:
                v = v.dict()
            result[k] = v
        return result
