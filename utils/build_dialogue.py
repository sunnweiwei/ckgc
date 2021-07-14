import sys
sys.path += ['./']

from utils.io import write_pkl, write_file, read_file
from tqdm import tqdm
import json

def dohash(text):
    o = ''
    for item in text:
        if item.isalpha():
            o += item
    return o


# for ckgc dataset
for lang in ['fr', 'es', 'zh']:
    data = json.load(open(f'dataset/ckgc/{lang}.json'))
    context = []
    response = []
    knowledge = ['no_passage_used']
    knowledge2id = {'no_passage_used': 0}
    pools = []
    for session in data:
        prefix = session['topic']
        for turn in session['dialogue']:
            if turn['role'] == 'Wizard':
                pool = []
                fg = 1
                for tt, ps in turn['knowledge_pool'].items():
                    for k in ps:
                        ks = tt + ' </s> ' + k
                        if ks not in knowledge2id:
                            knowledge2id[ks] = len(knowledge)
                            knowledge.append(ks)
                        if dohash(k.strip()) == dohash(turn['selected_knowledge'].strip()):
                            pool = [knowledge2id[ks]] + pool
                            fg = 0
                        else:
                            pool.append(knowledge2id[ks])
                if fg:
                    pool = [0] + pool
                if 0 not in pool:
                    pool.append(0)
                context.append(prefix)
                response.append(turn['text'])
                pools.append(pool)
            prefix = prefix + ' </s> ' + turn['text']
    write_file(context, f'dataset/ckgc/{lang}/context.txt')
    write_file(response, f'dataset/ckgc/{lang}/response.txt')
    write_file(knowledge, f'dataset/ckgc/{lang}/knowledge.txt')
    write_file(pools, f'dataset/ckgc/{lang}/pool.txt')

input('>>>>')
# for reddit-english
data = []
data.extend(read_file('large_raw/reddit_conversations.3turns.train.topical.txt'))
data.extend(read_file('large_raw/reddit_conversations.3turns.dev.topical.txt'))
data.extend(read_file('large_raw/reddit_conversations.3turns.test.topical.txt'))

context = []
response = []
for conv in tqdm(data):
    conv = conv.split('\t')
    for i in range(2):
        context.append(conv[i])
        response.append(conv[i+1])

write_file(context, f'dataset/reddit_en/context.txt')
write_file(response, f'dataset/reddit_en/response.txt')

# for tieba-chinese

data = read_file('raw/tieba.dialogues')
context = []
response = []

for line in data:
    ctx, res = line.split('\t')
    context.append(ctx)
    response.append(res)

with open('data/' + 'context.txt', 'w', encoding='utf-8') as f:
    for line in context:
        f.write(line + '\n')
with open('data/' + 'response.txt', 'w', encoding='utf-8') as f:
    for line in response:
        f.write(line + '\n')

