import sys

sys.path += ['./']

from utils.io import all_file, read_file, write_file, read_pkl, write_pkl
from utils.multiprocess import do_multiprocessing

from tqdm import tqdm
import os
import time
import pickle
import multiprocessing
import argparse
import logging


def get_tokenizer(name):
    if name == 'mbart':
        from transformers import MBartTokenizer
        return MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
    elif name == 'mbert':
        from transformers import BertTokenizer
        BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    from transformers import MBartTokenizer
    return MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")


def cut(inputs, method):
    if method == 'mbart':
        return inputs[:-1]
    elif method == 'mbert':
        return inputs[1:]
    else:
        return inputs


def tokenize(ids, data, method='mbart'):
    tokenizer = get_tokenizer(method)
    results = []
    for line in tqdm(data):
        results.append(sum([cut(tokenizer.encode(item), method) for item in line.split(' </s> ')], []))
    return ids, results


def tokenize_wiki(ids, step, method, input_path, output_path, size=1000000, sep=' </s> '):
    tokenizer = get_tokenizer(method)
    tk = iter(tqdm(range(1000000 * step), total=1000000 * step))
    for i in range(ids, ids + step):
        if not os.path.exists(f'{input_path}/{i}.txt'):
            continue
        data = []
        knowledge = [line[:-1].lower() for line in open(f'{input_path}/{i}.txt', encoding='utf-8')]
        for line in knowledge:
            next(tk)
            data.append(sum([cut(tokenizer.encode(item), method) for item in line.split(' </s> ')], []))
        write_pkl(data, f'{output_path}/{i}.pkl')


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str)
    parser.add_argument('-o', type=str)
    parser.add_argument('-t', type=str)
    parser.add_argument('-m', type=str)
    parser.add_argument('--p', type=int, default=1)
    parser.add_argument('--redis', type=int, default=1)
    args = parser.parse_args()

    input_path = args.i
    output_path = args.o
    task = args.t
    processes = args.p
    method = args.m
    use_redis = args.redis

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logging.info(f'Tokenize data, processes={processes}')

    if task == 'wiki':
        pool = multiprocessing.Pool(processes=processes)
        results = []
        file_num = len(all_file(input_path))
        step = file_num // processes
        for i in range(0, file_num, step):
            results.append(pool.apply_async(tokenize_wiki, (i, step, method, input_path, output_path)))
        pool.close()
        pool.join()

        if use_redis:
            import redis
            logging.info('Now build redis')
            data = []
            for i in range(1000):
                if os.path.exists(f'{output_path}/{i}.pkl'):
                    batch = read_pkl(f'{output_path}/{i}.pkl')
                    data.extend(batch)

            r = redis.StrictRedis(host='localhost', port=6379, db=0)
            pipe = r.pipeline()

            step = len(data) // 10
            for j, line in enumerate(data):
                key = str(j)
                value = pickle.dumps(line)
                pipe.set(key, value)
                if j % step == 0 and j != 0:
                    # print(j / len(data), 'execute')
                    pipe.execute()
            pipe.execute()
            # print('final execute done')
            # print('DONE!')

    else:
        context = [line[:-1].lower() for line in open(f'{input_path}/context.txt', encoding='utf-8')]
        context_ids = do_multiprocessing(tokenize, context, processes)
        write_pkl(context_ids, f'{output_path}/context.pkl')

        response = [line[:-1].lower() for line in open(f'{input_path}/response.txt', encoding='utf-8')]
        response_ids = do_multiprocessing(tokenize, response, processes)
        write_pkl(response_ids, f'{output_path}/response.pkl')

        if os.path.exists(f'{input_path}/knowledge.txt'):
            knowledge = [line[:-1].lower() for line in open(f'{input_path}/knowledge.txt', encoding='utf-8')]
            knowledge_ids = do_multiprocessing(tokenize, knowledge, processes)
            write_pkl(knowledge_ids, f'{output_path}/knowledge.pkl')


if __name__ == '__main__':
    main()
