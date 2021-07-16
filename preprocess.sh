!/bin/bash

cd dataset
if [ ! -d wiki  ];then
  mkdir wiki
fi

cd wiki
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
python -m wikiextractor.WikiExtractor enwiki-latest-pages-articles.xml.bz2
cd ..; cd ..
python utils/build_wiki.py -i dataset/wiki/text --o dataset/wiki/data --sep </s> --name wikipedia --language english

cd dataset
if [ ! -d reddit_en  ];then
  mkdir reddit_en
fi
wget https://drive.google.com/u/0/uc?export=download&confirm=rS6S&id=1T_ireYlvmMAqrAa9jfoUUw_9Njmc5NKF
tar zxvf reddit_conversations_v1.0_3turns.topical.tgz
cd ..;cd ..
python utils/build_dialogue.py

python utils/solr_retrieval.py -q dataset/reddit_en/response.txt -d wikipeida -o dataset/reddit_en/pool/pkl --p 40 --rows 32 --language english

python utils/tokenize_data.py -i dataset/ckgc/zh -o dataset/ckgc/zh -t reddit -m mbart --p 1 --redis 0
python utils/tokenize_data.py -i dataset/ckgc/fr -o dataset/ckgc/fr -t reddit -m mbart --p 1 --redis 0
python utils/tokenize_data.py -i dataset/ckgc/es -o dataset/ckgc/es -t reddit -m mbart --p 1 --redis 0
python utils/tokenize_data.py -i dataset/reddit_en -o dataset/reddit_en -t reddit -m mbart --p 20 --redis 0
python utils/tokenize_data.py -i dataset/wiki/data -o dataset/wiki/ids -t wiki -m mbart --p 40 --redis 1


