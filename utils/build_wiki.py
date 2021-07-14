from utils.io import all_file, read_file, write_file

from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import time
import os

import argparse
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str)
    parser.add_argument('--o', type=str, default='dataset/wiki/data')
    parser.add_argument('--size', type=int, default=1000000)
    parser.add_argument('--sep', type=str, default=' </s> ')
    parser.add_argument('--name', type=str, default='wikipedia')
    parser.add_argument('--language', type=str, default='english')
    args = parser.parse_args()

    input_path = args.i
    output_path = args.o
    size = args.size
    sep = args.sep
    name = args.name
    language = args.language

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    t = time.time()
    logging.info(f'Read WIKI data')
    passage = ['no_passage_used\nno_passage_used']
    doc = ''
    for file in tqdm(all_file(input_path)):
        data = read_file(file)
        for line in data:
            if '<doc id' in line:
                doc = doc[:-1]
                passage.append(doc)
                doc = ''
            else:
                if line != '' and line != '</doc>':
                    doc += line + '\n'

    logging.info(f'Cut WIKI to sentence')
    knowledge_collect = []
    collect_num = 0
    for p in tqdm(passage):
        k = p.split('\n')
        topic = k[0]
        article = ' '.join(k[1:])
        sentences = sent_tokenize(article, language=language)
        for s in sentences:
            knowledge = topic + sep + s
            knowledge_collect.append(knowledge)
            if len(knowledge_collect) == size:
                write_file(knowledge_collect, f'{output_path}/{collect_num}.txt')
                knowledge_collect = []
                collect_num += 1

    write_file(knowledge_collect, f'{output_path}/{collect_num}.txt')

    logging.info(f'Index WIKI in Solr')

    knowledge = []
    for i in range(1000):
        if os.path.exists(f'{output_path}/{i}.txt'):
            logging.info('Load file', f'{output_path}/{i}.txt', len(knowledge))
            knowledge.extend([line[:-1] for line in open(f'{output_path}/{i}.txt', encoding='utf-8')])

    import pysolr

    solr = pysolr.Solr(f'http://localhost:8983/solr/{name}/', always_commit=True)
    solr.ping()
    solr.delete(q='*:*')
    write = [
        {'kid': str(i),
         'content': str(line.strip())
         }
        for i, line in enumerate(knowledge)]
    solr.add(write)

    logging.info('Test result, query=harry potter, return=',
                 [dict(kid=res['kid'][0], content=res['content'][0])
                  for res in solr.search('content:harry&&potter', rows=2)])

    logging.info(f'Build WIKI corpus in {time.time() - t} sec')
