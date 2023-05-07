# Conversations Powered by Cross-Lingual Knowledge

We propose the cross-lingual knowledge grounded conversation (CKGC) task that ground open-domain dialogue by cross-lingual knowledge. We collect a test collection (see [dataset](https://github.com/sunnweiwei/ckgc/tree/main/dataset)), and propose a curriculum self-knowledge distillation scheme for CKGC.

## Data Format
The annotated data CKGC is at `/dataset`. An example from *zh.json*.

```
{
    'topic': 'Red meat',  // The start topic of the conversation
    'dialogue': [  // Conversation content
        {
            'role': 'Apprentice',  // The role of the speaker. Can be 'Apprentice' or 'Wizard'
            'text': '这个具体指哪些肉呢，日常生活中常见吗',  // The content of the message
            'knowledge_pool': [],  // Apprentice does not use knowledge
            'selected_knowledge': ""
        },
        {
            'role': 'Wizard',  // Now the speaker is Wizard.
            'text': '从营养学的角度来说，红肉一般含有更多肌红蛋白，比如牛肉啦',
            'knowledge_pool': {  // The candidate knowledge that wizard sees.
                'red meat':[  // Title of the article.
                    'red meat is a source of lipoic acid.',  // A sentence in the article.
                    'in nutritional science, "red meat" is defined as any meat that has more of the protein myoglobin than white meat.'
                    ...
                ],
                ...
            },
            'selected_knowledge': 'in nutritional science, "red meat" is defined as any meat that has more of the protein myoglobin than white meat.'  // The sentence selected by the Wizard. It can be a sentence in knowledge pool, or no_ passage_ used.
        },
        ...
    ]
}
```

The dataset is collected using annotation systems [test system](https://github.com/sunnweiwei/ckgc-system).

We are also collecting and sharing a [Multilingual Conversation Corpus](https://drive.google.com/drive/folders/1DJtWJKO7OGTHHLx2jkZNKUvfOK7l6rpN?usp=sharing) (in a uniform format), which includes 
- ~40M Reddit dialogue data of 36 languages, cleaned from 6 years of Reddit data, 
- ~0.8M personalized dialogue data of 7 languages, translated from persona dialogue, 
- ~0.3M knowledge-grounded dialogue data of 4 languages, translated from Wizard-of-Wikipedia, and 
- 12K CKGC test data of 4 languages, annotated in this project.

## Model
The code for reproducing the cross-lingual knowledge selection and response generation model are avaible at `modeling/train_retriever.py` and `modeling/train_generator.py`, respectively.


## Cite

```
@inproceedings{Sun:2021:CPC,
  author =    {Sun, Weiwei and Meng, Chuan and Meng, Qi and Ren, Zhaochun and Ren, Pengjie and Chen, Zhumin and de Rijke, Maarten},
  title =     {Conversations Powered by Cross-Lingual Knowledge},
  booktitle = {Proceedings of the 44rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  series =    {SIGIR '21},
  year =      {2021},
  publisher = {ACM}
}
```
