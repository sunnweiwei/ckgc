import os
import pickle


def all_file(dirname):
    fl = []
    for root, dirs, files in os.walk(dirname):
        for item in files:
            path = os.path.join(root, item)
            fl.append(path)
    return fl


def read_file(filename):
    with open(filename, encoding='utf-8') as f:
        return [line[:-1] for line in f]


def write_file(obj, filename):
    dirname = '/'.join(filename.split('/')[:-1])
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    with open(filename, 'w', encoding='utf-8') as f:
        for line in obj:
            f.write(str(line) + '\n')


def write_pkl(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def read_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
