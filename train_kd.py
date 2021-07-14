from modeling.models import Retriever, Generator, mbart_lang_to_id
from utils.io import read_pkl, write_pkl, write_file
from utils.data import Data, collate_fn, get_data_loader, CKGCTestData, collate_ckgc, remove_duplicates, DuoData

import torch
import torch.nn.functional as F
from transformers import AdamW
from tqdm import tqdm
import numpy as np
import time
import os
import argparse
import logging


def train_retriever(retriever, optimizer, dataset, pad_idx=1, batch_size=32, step=10):
    retriever.train()
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn(pad_idx),
        batch_size=batch_size,
        shuffle=False)
    label = torch.tensor([i for i in range(4096)]).long().cuda()
    tk0 = tqdm(data_loader, total=len(data_loader))
    item_num = 0
    acc_report = 0
    loss_report = 0
    now = 0
    for batch in tk0:
        if now >= step:
            break
        now += 1
        context = batch['context'].cuda()
        context2 = batch['context2'].cuda()
        knowledge_pool = batch['knowledge_pool'].cuda()
        # concat_context = batch['concat_context'].cuda()
        # response = batch['response'].cuda()
        # d_id = batch['d_id']
        # k_id = batch['k_id']

        bc_size = context.size(0)
        item_num += bc_size

        context = retriever(context)
        context2 = retriever(context2)
        knowledge = retriever(knowledge_pool)

        _, pooled_dim = context['pooled'].size()
        _, compressed_dim = context['compressed'].size()

        pooled_context = context['pooled'] / np.sqrt(pooled_dim)
        pooled_context2 = context2['pooled'] / np.sqrt(pooled_dim)
        pooled_knowledge = knowledge['pooled'] / np.sqrt(pooled_dim)
        attention = torch.mm(pooled_context, pooled_knowledge.t())
        attention2 = torch.mm(pooled_context2, pooled_knowledge.t())
        pooled_loss = F.kl_div(attention, attention2.detach()) + F.cross_entropy(attention2, label[:bc_size])
        # pooled_loss = F.cross_entropy(attention, label[:bc_size])

        compressed_context = context['compressed'] / np.sqrt(compressed_dim)
        compressed_context2 = context2['compressed'] / np.sqrt(compressed_dim)
        compressed_knowledge = knowledge['compressed'] / np.sqrt(compressed_dim)
        attention = torch.mm(compressed_context, compressed_knowledge.t())
        attention2 = torch.mm(compressed_context2, compressed_knowledge.t())
        compressed_loss = F.kl_div(attention, attention2.detach()) + F.cross_entropy(attention2, label[:bc_size])
        # compressed_loss = F.cross_entropy(attention, label[:bc_size])

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


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('-q1', type=str)
    parser.add_argument('-q2', type=str)
    parser.add_argument('-d', type=str)
    parser.add_argument('-pool', type=str)
    parser.add_argument('-m', type=int)
    parser.add_argument('-save_path', type=str)
    parser.add_argument('--bc_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--pool_size', type=int, default=10)
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--language', type=str, default='en')
    parser.add_argument('--pt_path', type=str, default='none')
    args = parser.parse_args()

    query1_path = args.q1
    query2_path = args.q2
    document_path = args.d
    pool_path = args.pool
    max_step = args.m
    batch_size = args.bc_size
    lr = args.lr
    pool_size = args.pool_size
    max_len = args.max_len
    lang_code = mbart_lang_to_id[args.language]
    save_path = args.save_path

    logging.info(f'Load query from {query1_path}-{query2_path} and document from {document_path}')

    query1 = read_pkl(query1_path)
    query2 = read_pkl(query2_path)
    if document_path != 'redis':
        document = []
        for i in range(200):
            if os.path.exists(f'{document_path}/{i}.pkl'):
                document.extend(read_pkl(f'{document_path}/{i}.pkl'))
    else:
        document = document_path
    knowledge_pool = read_pkl(pool_path)

    dataset = DuoData(query1, query1, query2, query2, knowledge_pool, pool_size=pool_size, knowledge=document,
                      order=None, max_len=max_len, lang_code=lang_code, curriculum=max_step)
    test_dataset = CKGCTestData(args.language, pool=f'dataset/ckgc/{args.language}/pool.txt',
                                max_len=max_len, lang_code=lang_code)

    logging.info('Build retriever')
    retriever = Retriever()
    if torch.cuda.is_available():
        retriever = retriever.cuda()

    optimizer = AdamW(retriever.parameters(), lr)
    pretrained_path = args.pt_path
    if os.path.exists(pretrained_path):
        logging.info(f'Load pretrained model from {pretrained_path}')
        retriever.load_state_dict({k.replace("module.", ""): v for k, v in torch.load(pretrained_path).items()})

    cur_step = 0
    while cur_step < max_step:
        dataset.set_offset(cur_step)
        logging.info(f'Training step {cur_step} / max step {max_step}')
        train_retriever(retriever, optimizer, dataset, pad_idx=1, batch_size=batch_size, step=10)
        cur_step += 10 * batch_size
        ranks = test_retriever(retriever, test_dataset, pad_idx=1, batch_size=batch_size, epoch=0)
        write_file(ranks, f'{save_path}/ranks/{cur_step}.txt')
        torch.save(retriever.state_dict(), f'{save_path}/retriever/{cur_step}.pt')


if __name__ == '__main__':
    main()
