from modeling.models import Retriever, Generator, mbart_lang_to_id
from utils.io import read_pkl, write_pkl, write_file
from utils.data import Data, collate_fn, get_data_loader, CKGCTestData, collate_ckgc
from utils.evaluation import eval_f1, f1_score, eval_rouge, eval_bleu, eval_meteor, eval_distinct, eval_all
from utils.tokenize_data import get_tokenizer

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


def train_generator(generator, optimizer, dataset, pad_idx=1, batch_size=32, epoch=0, distributed=True):
    generator.train()
    data_loader = get_data_loader(dataset, collate_fn(pad_idx), batch_size, distributed, epoch)
    tk0 = tqdm(data_loader, total=len(data_loader))
    item_num = 0
    acc_report = []
    ppl_report = 0
    for batch in tk0:
        # context = batch['context'].cuda()
        # knowledge_pool = batch['knowledge_pool'].cuda()
        concat_context = batch['concat_context'].cuda()
        response = batch['response'].cuda()
        # d_id = batch['d_id']
        # k_id = batch['k_id']

        predict = generator(concat_context, response)['logits']

        bc_size, length, emb_size = predict.size()
        predict = predict[:, :-1, :].contiguous().view(-1, emb_size)
        gt = response[:, 1:].contiguous().view(-1)
        length -= 1

        loss = F.cross_entropy(predict, gt, ignore_index=pad_idx, reduction='none')
        loss = loss.view(-1, length).sum(dim=1)
        tk_num = response.ne(pad_idx).long().sum(dim=-1)

        ppl_report += torch.exp(loss / tk_num).sum().item()
        item_num += bc_size
        acc_report.append(((predict.argmax(dim=-1) == gt) & gt.ne(pad_idx)).sum().item() / tk_num.sum().item())

        loss = loss.sum() / tk_num.sum()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(generator.parameters(), 2)

        optimizer.step()
        optimizer.zero_grad()

        tk0.set_postfix(acc=round(sum(acc_report) / len(acc_report), 3), ppl=ppl_report / item_num)


def test_generator(generator, dataset, language, tokenizer, pad_idx=1, batch_size=32, epoch=0, word_mask=None):
    generator.eval()
    data_loader = get_data_loader(dataset, collate_ckgc(pad_idx), batch_size, False, epoch)
    tk0 = tqdm(data_loader, total=len(data_loader))
    f1_report = []
    outputs_predict = []
    outputs_true = []
    with torch.no_grad():
        for batch in tk0:
            # context = batch['context'].cuda()
            # knowledge_pool = batch['knowledge_pool'].cuda()
            concat_context = batch['concat_context'].cuda()
            response = batch['response'].cuda()
            # d_ids = batch['d_id']
            # k_ids = batch['k_id']

            predict = generator.generate(input_ids=concat_context,
                                         decoder_start_token_id=tokenizer.lang_code_to_id[language],
                                         num_beams=3,
                                         max_length=128,
                                         bad_words_ids=word_mask)

            predict_sent = tokenizer.batch_decode(predict, skip_special_tokens=True)
            label_sent = tokenizer.batch_decode(response, skip_special_tokens=True)

            outputs_predict.extend(predict_sent)
            outputs_true.extend(label_sent)
            if language == 'zh':
                f1 = [f1_score(' '.join(pred), [' '.join(label)]) for pred, label in zip(predict_sent, label_sent)]
            else:
                f1 = [f1_score(pred, [label]) for pred, label in zip(predict_sent, label_sent)]
            f1_report.extend(f1)

            tk0.set_postfix(f1score=round(sum(f1_report) / len(f1_report), 4))

    return outputs_predict, outputs_true


def dist_init():
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('-dialog', type=str)
    parser.add_argument('-k', type=str)
    parser.add_argument('-pool', type=str)
    parser.add_argument('-save_path', type=str)
    parser.add_argument('--bc_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--pool_size', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--language', type=str, default='en')
    parser.add_argument('--pt_path', type=str, default='none')
    parser.add_argument('--dist', type=int, default=1)
    args = parser.parse_args()

    dialog_path = args.dialog
    knowledge_path = args.k
    pool_path = args.pool
    batch_size = args.bc_size
    lr = args.lr
    pool_size = args.pool_size
    max_len = args.max_len
    lang_code = mbart_lang_to_id[args.language]
    distributed = args.dist
    save_path = args.save_path
    language = args.language

    if distributed:
        dist_init()
    local_rank = dist.get_rank() if distributed else 0

    if knowledge_path != 'redis':
        knowledge = []
        for i in range(200):
            if os.path.exists(f'{knowledge_path}/{i}.pkl'):
                knowledge.extend(read_pkl(f'{knowledge_path}/{i}.pkl'))
    else:
        knowledge = knowledge_path
    knowledge_pool = read_pkl(pool_path)

    dataset = Data(read_pkl(f'{dialog_path}/context.pkl'),
                   read_pkl(f'{dialog_path}/response.pkl'),
                   knowledge_pool, pool_size=pool_size, knowledge=knowledge, order=None,
                   max_len=max_len, lang_code=lang_code)
    test_dataset = CKGCTestData(args.language, pool=f'dataset/ckgc/{args.language}/pool.txt',
                                max_len=max_len, lang_code=lang_code)

    tokenizer = get_tokenizer('mbart')
    tokenizer.lang_code_to_id = mbart_lang_to_id

    logging.info('Build generator')
    generator = Generator()
    if torch.cuda.is_available():
        generator = generator.cuda()
    if distributed:
        generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[local_rank],
                                                              output_device=local_rank, find_unused_parameters=True)
    optimizer = AdamW(generator.parameters(), lr)
    pretrained_path = args.pt_path
    if os.path.exists(pretrained_path):
        logging.info(f'Load pretrained model from {pretrained_path}')
        if distributed:
            dist.barrier()
            map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
            generator.load_state_dict(torch.load(pretrained_path, map_location=map_location))
            dist.barrier()
        else:
            generator.load_state_dict({k.replace("module.", ""): v for k, v in torch.load(pretrained_path).items()})

    for epoch in range(100):
        if os.path.exists(f'{save_path}/generator/{epoch}.pt'):
            if distributed:
                dist.barrier()
                map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
                generator.load_state_dict(torch.load(f'{save_path}/generator/{epoch}.pt', map_location=map_location))
                dist.barrier()
            else:
                generator.load_state_dict(
                    {k.replace("module.", ""): v for k, v in torch.load(save_path + f'_{epoch}.pt').items()})
            continue

        if distributed:
            dist.barrier()
        logging.info(f'Training epoch {epoch}')
        train_generator(generator, optimizer, dataset,
                        pad_idx=1, batch_size=batch_size, epoch=epoch, distributed=distributed)

        if distributed:
            dist.barrier()
        if local_rank == 0:
            predict, true = test_generator(generator, test_dataset, language, tokenizer,
                                           pad_idx=1, batch_size=batch_size, epoch=epoch, word_mask=None)
            logging.info(eval_all(predict, true))
            write_file(predict, f'{save_path}/predict/{epoch}.txt')
            torch.save(generator.state_dict(), f'{save_path}/generator/{epoch}.pt')
        if distributed:
            dist.barrier()


if __name__ == '__main__':
    main()
