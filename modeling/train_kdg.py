import sys

sys.path += ['./']

from modeling.models import Retriever, Generator, mbart_lang_to_id
from utils.io import read_pkl, write_pkl, write_file
from utils.data import Data, collate_fn, get_data_loader, CKGCTestData, collate_ckgc, DuoData
from utils.evaluation import eval_f1, f1_score, eval_rouge, eval_bleu, eval_meteor, eval_distinct, eval_all
from utils.tokenize_data import get_tokenizer

import torch
import torch.nn.functional as F
from transformers import AdamW
from tqdm import tqdm
import numpy as np
import time
import os
import argparse
import logging


def train_generator(generator, optimizer, dataset, pad_idx=1, batch_size=32, step=10):
    generator.train()
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn(pad_idx),
        batch_size=batch_size,
        shuffle=False)
    tk0 = tqdm(data_loader, total=len(data_loader))
    item_num = 0
    acc_report = []
    ppl_report = 0
    now = 0
    for batch in tk0:
        if now >= step:
            break
        now += 1
        # context = batch['context'].cuda()
        # knowledge_pool = batch['knowledge_pool'].cuda()
        concat_context = batch['concat_context'].cuda()
        response = batch['response'].cuda()
        concat_context2 = batch['concat_context2'].cuda()
        response2 = batch['response2'].cuda()
        # d_id = batch['d_id']
        # k_id = batch['k_id']
        content_part = batch['content_part'].ne(pad_idx).cuda()
        mask = content_part.ne(pad_idx)

        output = generator(concat_context, response)
        output2 = generator(concat_context2, response2)

        attention = output['decoder_attentions']
        attention2 = output2['decoder_attentions']

        min_len = min(concat_context.size(1), concat_context2.size(1))
        attention = torch.stack([item.mean(2)[:, :, :min_len].masked_fill(~mask, 0) for item in attention])
        attention2 = torch.stack([item.mean(2)[:, :, :min_len].masked_fill(~mask, 0) for item in attention2])
        attention = attention.view(-1, attention.size(-1))
        attention2 = attention2.view(-1, attention2.size(-1))

        kd = F.kl_div(attention, attention2, reduction='batchmean')

        predict = output['logits']
        predict2 = output2['logits']

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

        loss2 = F.cross_entropy(predict2, gt, ignore_index=pad_idx, reduction='none')

        final_loss = loss + loss2 + kd

        final_loss.backward()

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

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('-dialog', type=str)
    parser.add_argument('-dialog2', type=str)
    parser.add_argument('-k', type=str)
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

    dialog_path = args.dialog
    dialog2_path = args.dialog2
    knowledge_path = args.k
    pool_path = args.pool
    max_step = args.m
    batch_size = args.bc_size
    lr = args.lr
    pool_size = args.pool_size
    max_len = args.max_len
    lang_code = mbart_lang_to_id[args.language]
    save_path = args.save_path
    language = args.language

    if knowledge_path != 'redis':
        knowledge = []
        for i in range(200):
            if os.path.exists(f'{knowledge_path}/{i}.pkl'):
                knowledge.extend(read_pkl(f'{knowledge_path}/{i}.pkl'))
    else:
        knowledge = knowledge_path
    knowledge_pool = read_pkl(pool_path)

    dataset = DuoData(read_pkl(f'{dialog_path}/context.pkl'),
                      read_pkl(f'{dialog_path}/response.pkl'),
                      read_pkl(f'{dialog2_path}/context.pkl'),
                      read_pkl(f'{dialog2_path}/context.pkl'),
                      knowledge_pool, pool_size=pool_size, knowledge=knowledge, order=None,
                      max_len=max_len, lang_code=lang_code, curriculum=max_step)

    test_dataset = CKGCTestData(args.language, pool=f'dataset/ckgc/{args.language}/pool.txt',
                                max_len=max_len, lang_code=lang_code)

    tokenizer = get_tokenizer('mbart')
    tokenizer.lang_code_to_id = mbart_lang_to_id

    logging.info('Build generator')
    generator = Generator()
    if torch.cuda.is_available():
        generator = generator.cuda()

    optimizer = AdamW(generator.parameters(), lr)
    pretrained_path = args.pt_path
    if os.path.exists(pretrained_path):
        logging.info(f'Load pretrained model from {pretrained_path}')
        generator.load_state_dict({k.replace("module.", ""): v for k, v in torch.load(pretrained_path).items()})

    cur_step = 0
    while cur_step < max_step:
        dataset.set_offset(cur_step)
        logging.info(f'Training step {cur_step} / max step {max_step}')
        # train_generator(generator, optimizer, dataset,
        #                 pad_idx=1, batch_size=batch_size, step=10)
        cur_step += 10 * batch_size
        predict, true = test_generator(generator, test_dataset, language, tokenizer,
                                       pad_idx=1, batch_size=batch_size, epoch=0, word_mask=None)
        logging.info(eval_all(predict, true))
        write_file(predict, f'{save_path}/predict/{cur_step}.txt')
        torch.save(generator.state_dict(), f'{save_path}/generator/{cur_step}.pt')


if __name__ == '__main__':
    main()
