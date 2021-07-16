import sys

sys.path += ['./']

from modeling.models import Retriever, Generator, mbart_lang_to_id
from utils.io import read_pkl, write_pkl, write_file
from utils.data import Data, collate_fn, get_data_loader, CKGCTestData, collate_ckgc, remove_duplicates

import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AdamW
from tqdm import tqdm
import numpy as np
import time
import os
import argparse
import logging


def train_retriever(retriever, optimizer, dataset, pad_idx=1, batch_size=32, epoch=0, distributed=True):
    retriever.train()
    data_loader = get_data_loader(dataset, collate_fn(pad_idx), batch_size, distributed, epoch)
    label = torch.tensor([i for i in range(4096)]).long().cuda()
    tk0 = tqdm(data_loader, total=len(data_loader))
    item_num = 0
    acc_report = 0
    loss_report = 0
    for batch in tk0:
        context = batch['context'].cuda()
        knowledge_pool = batch['knowledge_pool'].cuda()
        # concat_context = batch['concat_context'].cuda()
        # response = batch['response'].cuda()
        # d_id = batch['d_id']
        # k_id = batch['k_id']

        bc_size = context.size(0)
        item_num += bc_size

        context = retriever(context)
        knowledge = retriever(knowledge_pool)

        _, pooled_dim = context['pooled'].size()
        _, compressed_dim = context['compressed'].size()

        pooled_context = context['pooled'] / np.sqrt(pooled_dim)
        pooled_knowledge = knowledge['pooled'] / np.sqrt(pooled_dim)
        attention = torch.mm(pooled_context, pooled_knowledge.t())
        pooled_loss = F.cross_entropy(attention, label[:bc_size])

        compressed_context = context['compressed'] / np.sqrt(compressed_dim)
        compressed_knowledge = knowledge['compressed'] / np.sqrt(compressed_dim)
        attention = torch.mm(compressed_context, compressed_knowledge.t())
        compressed_loss = F.cross_entropy(attention, label[:bc_size])

        loss = pooled_loss + compressed_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(retriever.parameters(), 2)
        optimizer.step()
        optimizer.zero_grad()

        acc_report += (attention.argmax(dim=-1) == label[:bc_size]).sum().item()
        loss_report += loss.item()

        tk0.set_postfix(loss=round(loss_report / item_num, 4), acc=round(acc_report / item_num, 4))


def test_retriever(retriever, dataset, pad_idx=1, batch_size=32, epoch=0):
    retriever.eval()
    data_loader = get_data_loader(dataset, collate_ckgc(pad_idx), batch_size, False, epoch)
    tk0 = tqdm(data_loader, total=len(data_loader))
    k_ranks = []
    rat1 = []
    rat5 = []
    with torch.no_grad():
        for batch in tk0:
            context = batch['context'].cuda()
            knowledge_pool = batch['knowledge_pool'].cuda()
            # concat_context = batch['concat_context'].cuda()
            # response = batch['response'].cuda()
            # d_ids = batch['d_id']
            k_ids = batch['k_id']

            context = retriever(context)['pooled']

            bc_size, pool_size, _ = knowledge_pool.size()
            knowledge_pool = knowledge_pool.view(bc_size * pool_size, -1)
            knowledge_pool = retriever(knowledge_pool)['pooled']
            knowledge_pool = knowledge_pool.view(bc_size, pool_size, -1)

            d_model = context.size(-1)
            context = context / np.sqrt(d_model)
            knowledge_pool = knowledge_pool / np.sqrt(d_model)

            attention = torch.bmm(knowledge_pool, context.unsqueeze(-1)).squeeze(-1)

            # chose = attention.argmax(dim=-1)

            rank = [remove_duplicates([kid[ri] for ri in prob.argsort()]) for prob, kid in zip(attention, k_ids)]
            k_ranks.extend(rank)

            rat1.extend([int(true[0] == pred[0]) for true, pred in zip(k_ids, rank)])
            rat5.extend([int(true[0] in pred[:5]) for true, pred in zip(k_ids, rank)])

            tk0.set_postfix(rat1=round(sum(rat1) / len(rat1), 4), rat5=round(sum(rat5) / len(rat5), 4))

    return k_ranks


def dist_init():
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('-q', type=str)
    parser.add_argument('-d', type=str)
    parser.add_argument('-pool', type=str)
    parser.add_argument('-save_path', type=str)
    parser.add_argument('--bc_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--pool_size', type=int, default=10)
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--language', type=str, default='en')
    parser.add_argument('--pt_path', type=str, default='none')
    parser.add_argument('--dist', type=int, default=1)
    args = parser.parse_args()

    query_path = args.q
    document_path = args.d
    pool_path = args.pool
    batch_size = args.bc_size
    lr = args.lr
    pool_size = args.pool_size
    max_len = args.max_len
    lang_code = mbart_lang_to_id[args.language]
    distributed = args.dist
    save_path = args.save_path

    if distributed:
        dist_init()
    local_rank = dist.get_rank() if distributed else 0

    logging.info(f'Load query from {query_path} and document from {document_path}')

    query = read_pkl(query_path)
    if document_path != 'redis':
        document = []
        for i in range(200):
            if os.path.exists(f'{document_path}/{i}.pkl'):
                document.extend(read_pkl(f'{document_path}/{i}.pkl'))
    else:
        document = document_path
    knowledge_pool = read_pkl(pool_path)

    dataset = Data(query, query, knowledge_pool, pool_size=pool_size, knowledge=document, order=None,
                   max_len=max_len, lang_code=lang_code)
    test_dataset = CKGCTestData(args.language, pool=f'dataset/ckgc/{args.language}/pool.txt',
                                max_len=max_len, lang_code=lang_code)

    logging.info('Build retriever')
    retriever = Retriever()
    if torch.cuda.is_available():
        retriever = retriever.cuda()
    if distributed:
        retriever = torch.nn.parallel.DistributedDataParallel(retriever, device_ids=[local_rank],
                                                              output_device=local_rank, find_unused_parameters=True)
    optimizer = AdamW(retriever.parameters(), lr)
    pretrained_path = args.pt_path
    if os.path.exists(pretrained_path):
        logging.info(f'Load pretrained model from {pretrained_path}')
        if distributed:
            dist.barrier()
            map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
            retriever.load_state_dict(torch.load(pretrained_path, map_location=map_location))
            dist.barrier()
        else:
            retriever.load_state_dict({k.replace("module.", ""): v for k, v in torch.load(pretrained_path).items()})

    for epoch in range(100):
        if os.path.exists(f'{save_path}/retriever/{epoch}.pt'):
            if distributed:
                dist.barrier()
                map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
                retriever.load_state_dict(torch.load(f'{save_path}/retriever/{epoch}.pt', map_location=map_location))
                dist.barrier()
            else:
                retriever.load_state_dict(
                    {k.replace("module.", ""): v for k, v in torch.load(save_path + f'_{epoch}.pt').items()})
            continue

        if distributed:
            dist.barrier()
        logging.info(f'Training epoch {epoch}')
        train_retriever(retriever, optimizer, dataset,
                        pad_idx=1, batch_size=batch_size, epoch=epoch, distributed=distributed)

        if distributed:
            dist.barrier()
        if local_rank == 0:
            ranks = test_retriever(retriever, test_dataset, pad_idx=1, batch_size=batch_size, epoch=epoch)
            write_file(ranks, f'{save_path}/ranks/{epoch}.txt')
            torch.save(retriever.state_dict(), f'{save_path}/retriever/{epoch}.pt')
        if distributed:
            dist.barrier()


if __name__ == '__main__':
    main()
