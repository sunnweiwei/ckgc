from utils.multiprocess import do_multiprocessing
from utils.io import write_pkl

from tqdm import tqdm
import time
import argparse
import logging


def solr_client(ids, qs, name, rows):
    import pysolr
    solr = pysolr.Solr(f'http://localhost:8983/solr/{name}/', always_commit=True)
    solr.ping()
    tk = tqdm(qs, total=len(qs))
    data = []
    for q in tk:
        if q == 'content:no_passage_used':
            data.append([0])
        else:
            doc_id = [r['kid'][0] for r in solr.search(q, rows=rows)]
            data.append(doc_id)
    return ids, data


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('-q', type=str)
    parser.add_argument('-d', type=str)
    parser.add_argument('-o', type=str)
    parser.add_argument('--p', type=int, default=1)
    parser.add_argument('--rows', type=int, default=32)
    parser.add_argument('--language', type=str, default='english')
    args = parser.parse_args()

    query_path = args.q
    document_name = args.d
    output_path = args.o
    processes = args.p
    rows = args.rows
    language = args.language

    logging.info(f'Solr retrieval')

    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words(language))

    queries = [line[:-1].lower() for line in open(query_path, encoding='utf-8')]

    q4search = []
    for query in queries:
        query = '&&'.join(
            list(set([x for x in ''.join(x if x.isalnum() else ' ' for x in query).split() if x not in STOPWORDS])))
        if len(query) == 0:
            query = 'no_passage_used'
        query = 'content:' + query
        q4search.append(query)

    import multiprocessing

    pool = multiprocessing.Pool(processes=processes)
    results = []
    length = len(q4search) // processes + 1
    for i in range(processes):
        collect = q4search[i * length:(i + 1) * length]
        results.append(pool.apply_async(solr_client, (i, collect, document_name, rows)))
    pool.close()
    pool.join()
    k = []
    for j, res in enumerate(results):
        ids, data = res.get()
        assert j == ids
        k.extend(data)

    write_pkl(output_path, k)


if __name__ == '__main__':
    main()
