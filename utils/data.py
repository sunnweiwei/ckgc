from abc import ABC
from utils.io import read_pkl, read_file

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import redis
import torch
import pickle
import numpy as np


class RedisData:
    def __init__(self):
        self.redis = redis.StrictRedis(host='localhost', port=6379, db=0)

    def __getitem__(self, item):
        return pickle.loads(self.redis.get(str(item)))


class Data(Dataset):
    def __init__(self, context, response, pool, pool_size=32, knowledge=None, order=None,
                 max_len=64, lang_code=250004, curriculum=False):
        super(Dataset, self).__init__()
        self.context = context
        self.response = response
        self.pool = pool
        self.pool_size = pool_size
        self.curriculum = int(curriculum)

        if knowledge == 'redis':
            self.knowledge = RedisData()
        else:
            self.knowledge = knowledge

        self.order = order if order is not None else [j for j in range(len(self.context))]

        self.max_len = max_len
        self.lang_code = lang_code

    def __getitem__(self, index):
        if self.curriculum:
            index = np.random.randint(0, max(len(self.context) *
                                             min(1, (index * 0.99 / self.curriculum + 0.01) ** 0.5), 1))
        index = self.order[index]
        conv_his = self.context[index][:self.max_len]
        pool_idx = self.pool[index][:self.pool_size]

        knowledge_pool = []
        concat_context = []
        response = []
        d_id = []
        k_id = []
        for idx in pool_idx:
            d_id.append(index)
            k_id.append(int(idx))
            content = self.knowledge[int(idx)][:self.max_len]
            knowledge_pool.append(torch.tensor(content))
            concat_context.append(torch.tensor(content + conv_his))
            response.append(torch.tensor([self.lang_code] + self.response[index][:self.max_len]))
        return torch.tensor(conv_his), knowledge_pool, concat_context, response, d_id, k_id

    def __len__(self):
        if self.curriculum:
            return self.curriculum
        return len(self.order)


class DuoData(Data):
    def __init__(self, context, response, context2, response2, pool, pool_size=32, knowledge=None, order=None,
                 max_len=64, lang_code=250004, curriculum=False):
        super().__init__(context, response, pool, pool_size, knowledge, order,
                         max_len, lang_code, curriculum)
        self.context2 = context2
        self.response2 = response2
        self.offset = 0

    def __getitem__(self, index):
        index = index + self.offset
        conv_his, knowledge_pool, concat_context, response, d_id, k_id = super().__getitem__(index)
        index = d_id[0]
        conv_his2 = self.context2[index][:self.max_len]
        concat_context2 = []
        response2 = []
        content_part = []
        for idx in k_id:
            content = self.knowledge[int(idx)][:self.max_len]
            content_part.append(torch.tensor(content))
            concat_context2.append(torch.tensor(content + conv_his2))
            response2.append(torch.tensor([self.lang_code] + self.response[index][:self.max_len]))
        return conv_his, knowledge_pool, concat_context, response, d_id, k_id, \
               torch.tensor(conv_his2), concat_context2, response2, content_part

    def set_offset(self, offset):
        self.offset = offset


def flat_array(array):
    size = max([len(line) for line in array])
    return [line[i] for i in range(size) for line in array if i < len(line)]


def collate_fn(pad_idx):
    def f(data):
        context, knowledge_pool, concat_context, response, d_id, k_id, *auxiliary = zip(*data)

        context = context
        knowledge_pool = flat_array(knowledge_pool)
        concat_context = flat_array(concat_context)
        response = flat_array(response)
        d_id = flat_array(d_id)
        k_id = flat_array(k_id)

        collect = dict(
            context=pad_sequence(context, batch_first=True, padding_value=pad_idx),
            knowledge_pool=pad_sequence(knowledge_pool, batch_first=True, padding_value=pad_idx),
            concat_context=pad_sequence(concat_context, batch_first=True, padding_value=pad_idx),
            response=pad_sequence(response, batch_first=True, padding_value=pad_idx),
            d_id=d_id,
            k_id=k_id
        )

        if len(auxiliary) == 4:
            context2, concat_context2, response2, content_part = auxiliary
            context2 = context2
            concat_context2 = flat_array(concat_context2)
            response2 = flat_array(response2)
            content_part = flat_array(content_part)

            collect['context2'] = pad_sequence(context2, batch_first=True, padding_value=pad_idx)
            collect['concat_context2'] = pad_sequence(concat_context2, batch_first=True, padding_value=pad_idx)
            collect['response2'] = pad_sequence(response2, batch_first=True, padding_value=pad_idx)
            content_part[0] = torch.cat((content_part[0], torch.tensor([pad_idx] * (
                        min(collect['concat_context'].size(1), collect['concat_context2'].size(1)) - len(content_part[0])
            ))), 0)
            collect['content_part'] = pad_sequence(content_part, batch_first=True, padding_value=pad_idx)

        return collect

    return f


def get_data_loader(dataset, collate_func, batch_size, distributed, epoch):
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        data_sampler = DistributedSampler(dataset)
        data_sampler.set_epoch(epoch)
        loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_func,
            batch_size=batch_size,
            sampler=data_sampler)
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_func,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4)
    return loader


class CKGCTestData(Dataset, ABC):
    def __init__(self, lang, pool='pool', max_len=64, lang_code=250004):
        self.context = read_pkl(f'dataset/ckgc/{lang}/context.pkl')
        self.response = read_pkl(f'dataset/ckgc/{lang}/response.pkl')
        self.knowledge = read_pkl(f'dataset/ckgc/{lang}/knowledge.pkl')
        self.pool = [[int(item) for item in line[1:-1].split(',')] for line in read_file(pool)]

        self.max_len = max_len
        self.lang_code = lang_code

    def __getitem__(self, index):
        conv_his = self.context[index][-self.max_len:]
        response = torch.tensor([self.lang_code] + self.response[index][:self.max_len])

        knowledge_pool = [torch.tensor(self.knowledge[ids]) for ids in self.pool[index]]
        concat_context = torch.tensor(self.knowledge[self.pool[index][0]] + conv_his)

        d_id = index
        k_id = self.pool[index]

        return torch.tensor(conv_his), knowledge_pool, concat_context, response, d_id, k_id

    def __len__(self):
        return len(self.context)


def pad_array(array):
    size = max([len(line) for line in array])
    return [line + [line[-1]] * (size - len(line)) for line in array]


def remove_duplicates(array):
    outputs = []
    for item in array:
        if item not in outputs:
            outputs.append(item)
    return outputs


def collate_ckgc(pad_idx):
    def f(data):
        context, knowledge_pool, concat_context, response, d_id, k_id = zip(*data)

        context = context
        knowledge_pool = pad_array(knowledge_pool)
        pool_size = len(knowledge_pool[0])
        knowledge_pool = pad_sequence(sum(knowledge_pool, []), batch_first=True, padding_value=pad_idx).view(
            len(context), pool_size, -1)
        # knowledge_pool = [pad_sequence(line, batch_first=True, padding_value=pad_idx)
        #                   for line in pad_array(knowledge_pool)]
        concat_context = concat_context
        response = response
        d_id = d_id
        k_id = pad_array(k_id)
        return dict(
            context=pad_sequence(context, batch_first=True, padding_value=pad_idx),
            knowledge_pool=knowledge_pool,
            concat_context=pad_sequence(concat_context, batch_first=True, padding_value=pad_idx),
            response=pad_sequence(response, batch_first=True, padding_value=pad_idx),
            d_id=d_id,
            k_id=k_id
        )

    return f
