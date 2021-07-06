from .modeling_mbart import MBartForConditionalGeneration

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, MBartTokenizer

mbart_lang_to_id = {'ar_AR': 250001, 'cs_CZ': 250002, 'de_DE': 250003, 'en_XX': 250004, 'es_XX': 250005,
                    'et_EE': 250006, 'fi_FI': 250007, 'fr_XX': 250008, 'gu_IN': 250009, 'hi_IN': 250010,
                    'it_IT': 250011, 'ja_XX': 250012, 'kk_KZ': 250013, 'ko_KR': 250014, 'lt_LT': 250015,
                    'lv_LV': 250016, 'my_MM': 250017, 'ne_NP': 250018, 'nl_XX': 250019, 'ro_RO': 250020,
                    'ru_RU': 250021, 'si_LK': 250022, 'tr_TR': 250023, 'vi_VN': 250024, 'zh_CN': 250025}

mbart_lang_to_id = {k[:2]: v for k, v in mbart_lang_to_id.items()}


def universal_sentence_embedding(sentences, mask, sqrt=True):
    """
    Perform Universal Sentence Encoder averaging (https://arxiv.org/abs/1803.11175).

    This is really just sum / sqrt(len).

    :param Tensor sentences: an N x T x D of Transformer outputs. Note this is
        the exact output of TransformerEncoder, but has the time axis first
    :param ByteTensor: an N x T binary matrix of paddings

    :return: an N x D matrix of sentence embeddings
    :rtype Tensor:
    """
    # need to mask out the padded chars
    sentence_sums = torch.bmm(
        sentences.permute(0, 2, 1), mask.float().unsqueeze(-1)
    ).squeeze(-1)
    divisor = mask.sum(dim=1).view(-1, 1).float()
    if sqrt:
        divisor = divisor.sqrt()
    sentence_sums /= divisor
    return sentence_sums


class Retriever(nn.Module):
    def __init__(self, pad_idx=1, lang_code_to_id=None):
        super().__init__()
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")

        self.encoder = model.get_encoder()
        self.fc = nn.Linear(1024, 64)
        self.lang_code_to_id = lang_code_to_id
        self.pad_idx = pad_idx
        torch.nn.init.normal_(self.fc.weight, std=0.02)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_idx).long().detach()
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask,
                           output_attentions=False,
                           output_hidden_states=True,
                           return_dict=True)
        hidden_states = out.hidden_states
        pooled = universal_sentence_embedding(hidden_states[-1], attention_mask)
        fc_pooled = self.fc(pooled)
        return dict(
            pooled=pooled,
            compressed=fc_pooled
        )


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
        self.pad_idx = 1

        # checkpoint = 'project/criss/criss_checkpoints/criss.3rd.pt'
        # criss = torch.load(checkpoint)
        # mbart_state = self.model.state_dict()
        # criss_state = {k: criss[k[6:]] if k[6:] in criss else mbart_state[k] for k in mbart_state}
        # self.model.load_state_dict(criss_state)

    def encode(self, input_ids, attention_mask=None, **kwargs):
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_idx).long().detach()
        encoder_outputs, encoder_states = self.model.model.encoder(input_ids,
                                                                   attention_mask=attention_mask,
                                                                   output_attentions=False,
                                                                   output_hidden_states=True,
                                                                   return_dict=False)
        return dict(
            encoder_outputs=encoder_outputs,
            hidden_states=encoder_states,
            attention_mask=attention_mask
        )

    def decode(self, decoder_input_ids, encoder_outputs, **kwargs):
        attention_mask = encoder_outputs['attention_mask']
        encoder_outputs = [encoder_outputs['encoder_outputs']]
        out = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=None,
            encoder_outputs=encoder_outputs,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True
        )
        return dict(
            logits=out.logits,
            encoder_outputs=encoder_outputs,
            decoder_attentions=out.decoder_attentions
        )

    def forward(self, context, response, **kwargs):
        context = context
        c = self.encode(context, **kwargs)
        r = self.decode(response, c, **kwargs)
        return r

    def generate(
            self,
            input_ids,
            decoder_start_token_id,
            max_length=64,
            do_sample=False,
            num_beams=1,
            bad_words_ids=None
    ):
        pad_token_id = self.pad_idx
        out = self.model.generate(
            input_ids=input_ids,
            decoder_start_token_id=decoder_start_token_id,
            max_length=max_length,
            do_sample=do_sample,
            num_beams=num_beams,
            pad_token_id=pad_token_id,
            eos_token_id=None,
            bad_words_ids=bad_words_ids
        )

        return out


# all special tokens
# <s> 0            ar_AR 250001    et_EE 250006    it_IT 250011    lv_LV 250016    ru_RU 250021
# </s> 2           cs_CZ 250002    fi_FI 250007    ja_XX 250012    my_MM 250017    si_LK 250022
# <unk> 3          de_DE 250003    fr_XX 250008    kk_KZ 250013    ne_NP 250018    tr_TR 250023
# <pad> 1          en_XX 250004    gu_IN 250009    ko_KR 250014    nl_XX 250019    vi_VN 250024
# <mask> 250026    es_XX 250005    hi_IN 250010    lt_LT 250015    ro_RO 250020    zh_CN 250025

def main():
    print('hello')
    model = Generator()
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
    TEXT = 'This is a test'
    ids = tokenizer.encode(TEXT)
    outputs = model.generate(input_ids=torch.tensor([ids]), decoder_start_token_id=250004)
    print(tokenizer.batch_decode(outputs))
    # print(outputs)


if __name__ == '__main__':
    main()
