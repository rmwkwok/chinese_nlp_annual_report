# Pre-requisite
# =============================================================================
# 1. clone the repo (https://github.com/Jacen789/relation-extraction) under
#    this directory
# 2. save "pytorch_model.bin" from https://huggingface.co/bert-base-chinese 
#    under pretrained_models/bert-base-chinese/ of the repo
# 3. train a model according to the repo's instruction (~3hr with GPU)

# Note:
# =============================================================================
# 1. Available relationships: 父母, 夫妻, 师生, 兄弟姐妹, 合作, 情侣, 祖孙, 好友, 
#    亲戚, 同门, 上下级
# 2. Relationships regroupped as tag2grp

tag2grp = {
    '父母': 'Family',
    '夫妻': 'Family',
    '兄弟姐妹': 'Family',
    '祖孙': 'Family',
    '亲戚': 'Family',
    
    '师生': 'Work',
    '合作': 'Work',
    '同门': 'Work',
    '上下级': 'Work',
    
    '情侣': 'Friend',
    '好友': 'Friend',
    
    'unknown': 'Unknown'
}

import sys
sys.path.insert(1, './relation-extraction')

import torch
import warnings

import numpy as np

# don't change this unless you made relevant change in training's parameter
class hparams:
    # keeping only necessary attributes
    dropout=0.1
    embedding_dim=768 
    pretrained_model_path='./relation-extraction/pretrained_models/bert-base-chinese' 
    
    device='cpu'  
    model_file='./relation-extraction/saved_models/model.bin' 
    tagset_file='./relation-extraction/datasets/relation.txt' 
    
try:
    from relation_extraction.model import SentenceRE
    from relation_extraction.data_utils import MyTokenizer,\
                                               get_idx2tag,\
                                               convert_pos_to_mask

    # Get the mapping between idx and tag ready
    idx2tag = get_idx2tag(hparams.tagset_file)
    hparams.tagset_size = len(idx2tag)

    # get model ready
    model = SentenceRE(hparams).to(hparams.device)
    if hparams.device == 'cpu':
        model.load_state_dict(torch.load(
            hparams.model_file, 
            map_location=torch.device('cpu'),
        ))
    else:
        model.load_state_dict(torch.load(hparams.model_file))
    
    # set model in eval model
    model.eval()

    # get tokenizer ready
    tokenizer = MyTokenizer(hparams.pretrained_model_path)
except Exception as e:
    print(e)
    model = None
    idx2tag = None
    tokenizer = None
    warnings.warn('Cannot load Relation Extraction model. Did you train it?')
    
def tokenize_item(item):
    '''Tokenize an `item`.

    Keyword arguments:
    item -- `Dict`. Formatted as 
            {
                'text': '...', 
                'h':{'name':'...', 'pos':(x, y)},
                't':{'name':'...', 'pos':(x, y)},
            }, where h and t are the two terms of which a relationship is being
            looked for. pos is a tuple of 2 integers for the start and end 
            positions of the term. end-start = len(name).
    
    Outputs:
    model_input -- `Tuple`. As required by the model to infer.
    '''
    
    # tokenize
    tokens, pos_e1, pos_e2 = tokenizer.tokenize(item)
    encoded = tokenizer.bert_tokenizer.batch_encode_plus(
        [(tokens, None)], 
        return_tensors='pt',
    )

    # get parameters ready for model prediction
    input_ids = encoded['input_ids'].to(hparams.device)
    token_type_ids = encoded['token_type_ids'].to(hparams.device)
    attention_mask = encoded['attention_mask'].to(hparams.device)
    
    e1_mask = torch.tensor([
        convert_pos_to_mask(pos_e1, max_len=attention_mask.shape[1])
    ]).to(hparams.device)
    e2_mask = torch.tensor([
        convert_pos_to_mask(pos_e2, max_len=attention_mask.shape[1])
    ]).to(hparams.device)

    model_input = (input_ids, token_type_ids, attention_mask, e1_mask, e2_mask)
    
    return model_input

def predict(model_input):
    '''Predict entity relations.

    Keyword arguments:
    model_input -- as produced by `tokenize_item(...)`
    
    Outputs:
    logits -- `Tensor`.
    '''
    
    with torch.no_grad():
        logits = model(*model_input)[0]

    return logits

def get_proba(logits, mode='GROUPED_SUM'):
    '''Converting logits into probability.

    Keyword arguments:
    logits -- `Tensor`. List of logits.
    mode -- `str`. How to process the probabilites. 
            'GROUPED_SUM': group according to `tag2grp` and sum the 
                           probabilities 
            'GROUPED_MAX': group according to `tag2grp` and get the 
                           maximum probability value
    
    Outputs:
    probas -- `List` of `Tuple`. Sorted list of (probability, group name)
    '''
    
    logits = logits.numpy()
    probas = np.exp(logits)/sum(np.exp(logits))
    if mode == 'GROUPED_SUM':
        labels = np.array(list(map(tag2grp.get, 
                                   map(idx2tag.get, range(len(idx2tag))))))
        probas = [(probas[labels == l].sum(), l) for l in np.unique(labels)]
        
    elif mode == 'GROUPED_MAX':
        labels = np.array(list(map(tag2grp.get, 
                                   map(idx2tag.get, range(len(idx2tag))))))
        probas = [(probas[labels == l].max(), l) for l in np.unique(labels)]
        
    else:
        raise NotImplementedError
        
    return sorted(probas, reverse=True)

def gen_item(doc, h, t, max_length=512):
    '''Generate an `item` formatted as described in `tokenize_item(...)`. This
        function removes characters from the text if the text is too long. This
        is to cope with the length requirement set by the model, which is not
        ideal. The proper way would be to train a model which can accept 
        all required length of texts.

    Keyword arguments:
    doc -- `Dict`. 
    h -- `Dict`. The Head NER entity.
    t -- `Dict`. The Tail NER entity.
    max_length -- `Int`. Above which it will try to remove the excess amount of
                  characters from either the left or the right, depending on 
                  which side has more room, and as long as it will not remove 
                  the Head and the Tail entity. If there is no feasible side
                  to remove characters from, it does not remove anything.
    
    Outputs:
    model_input -- `Tuple`. As required by the model to infer.
    '''
    
    item = dict(
        text=doc['text'], 
        h=dict(name=h['text'], pos=(h['start'], h['end'])),
        t=dict(name=t['text'], pos=(t['start'], t['end'])),
    )
    
    n_strip = len(doc['text']) - max_length
    
    if n_strip > 0:
        
        left_room = min(h['start'], t['start']) - 1
        right_room = len(doc['text']) - max(h['end'], t['end'])
        
        # strip from left
        if left_room >= right_room and left_room > n_strip:
            item['text'] = item['text'][n_strip:]
            item['h']['pos'] = tuple(map(lambda x: x-1, item['h']['pos']))
        
        # strip from right
        elif right_room > left_room and right_room > n_strip:
            item['text'] = item['text'][:-n_strip]
        
        # do nothing
        else:
            print('Give up stripping')
            pass

    return item