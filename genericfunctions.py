import random as rand


def dict_to_str(dic):
    s = ""
    for value in dic.values():
        s += (str(value) + "/")
    return s


def func_all(a):
    def h(x):
        return all([f(x) for f in a])
    return h


def max_index(seq):
    max_value = seq[0]
    max_indexes = [0]

    for i in range(1, len(seq)):
        if seq[i] > max_value:
            max_value = seq[i]
            max_indexes = [i]
        elif seq[i] == max_value:
            max_indexes.append(i)
    return rand.choice(max_indexes)


def max_key(dic):
    values = list(dic.values())
    index = max_index(values)
    return list(dic.keys())[index]
