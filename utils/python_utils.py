from functools import reduce

def rget(d:dict, *keys):
    return reduce(lambda c, k: c.get(k, {}), keys, d)

def rset(d:dict, *keys, val):
    if len(keys) == 1:
        d[keys[0]] = val
    else:
        d.setdefault(keys[0],{})
        rset(d[keys[0]], *keys[1:], val=val)

# dataset = dict()
# rset(dataset,'l','train','X', val=1)
# print(dataset)
