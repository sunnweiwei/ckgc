## Conversations Powered by Cross-Lingual Knowledge

### Data Format

An example from *zh.json*.

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



We use the [test system](https://github.com/sunnweiwei/ckgc-system) to collect the dataset.

