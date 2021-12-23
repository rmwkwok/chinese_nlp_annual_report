# Pre-requisite
# =============================================================================
# 1. clone the repo (https://github.com/zerohd4869/Chinese-NER) under this 
#    directory
# 2. save "ctb.50d.vec" and "gigaword_chn.all.a2b.uni.ite50.vec" from 
#    https://awesomeopensource.com/project/jiesutd/LatticeLSTM under data/ of 
#    the repo
# 3. train a model according to the repo's instruction (~1.5 hr with CPU)
# 4. pick a model for inference and put its name into the "model_dir" variable

# Note:
# =============================================================================
# 1. The repo only provided example on accepting a file as inference's input,
#    this code will follow this practice in order to minimize unnecessary work
# 2. Available entity tag
#    CONT: Country
#    EDU: Educational Institution
#    LOC: Location
#    NAME: Personal Name
#    ORG: Organization
#    PRO: Profession
#    RACE: Ethnicity Background
#    TITLE: Job Title

import sys
sys.path.insert(1, './Chinese-NER')
import torch
import pickle
import warnings
import itertools

# change this
model_dir = './Chinese-NER/data/resume/model_wclstm_0210_batch1/saved_model-16-92.9.model'

# don't change this unless you made relevant change in training's parameter
dset_dir = './Chinese-NER/data/resume/model_wclstm_0210_batch1/saved_model.dset'

# load trained model data and the model
try:
    from model.cw_ner.lw.cw_ner import CW_NER
    from main import batchify_with_label_3, recover_label
    from utils.functions import read_instance_with_gaz_3

    with open(dset_dir, 'rb') as fp:
        model_data = pickle.load(fp)

    model_data.HP_gpu = False
    model_data.MAX_SENTENCE_LENGTH = 2300

    model = CW_NER(model_data)
    model.load_state_dict(torch.load(model_dir))

    # set model in eval model
    model.eval()
except:
    model = None
    warnings.warn('Cannot load NER model. Did you train it?')

def get_compatible_input(corpus):
    '''`corpus` will be converted into a file formatted to resemble the repo's 
        "data/resume/test.char.bmes", before being read again as instances that 
        is acceptable by the model.

    Keyword arguments:
    corpus -- `Dict`. Formatted as CTIL provided.
    
    Outputs:
    texts -- `List`. List of lists of tokens output by the model. It is used to
                reconstruct the document content.
    instances -- Compatible format for model input.
    '''
    
    predict_file_path = 'ner_input.temp'
    
    with open(predict_file_path, 'w') as f:
        for stock in corpus:
            for doc in stock['text']:
                f.write(' O\n'.join(doc['text']+'\n'))
                
    texts, instances = read_instance_with_gaz_3(
        predict_file_path, 
        model_data.gaz, model_data.char_alphabet, model_data.bichar_alphabet,
        model_data.gaz_alphabet, model_data.label_alphabet, 
        model_data.number_normalized, model_data.MAX_SENTENCE_LENGTH, 
        model_data.use_single
    )
    
    return texts, instances
    
def predict(instances):
    '''Predict the entities.

    Keyword arguments:
    instances -- as produced by `get_compatible_input(...)`
    
    Outputs:
    ner_predictions -- `List`. List of prediction for each of the `instances`. 
                        In each prediction, it has a list of entity labels as 
                        many as the number of words in that instance. 
                        Unidentified word is labelled as O. 
    '''
    
    # predict batch-by-batch
    batch_size = model_data.HP_batch_size
    ner_predictions = []

    for i in range(0, len(instances), batch_size):
        instance = instances[i: i+batch_size]

        # build batched data
        gaz_list, reverse_gaz_list, batch_char, batch_bichar, batch_charlen,\
            batch_charrecover, batch_label, mask =\
            batchify_with_label_3(
                instance, 
                model_data.HP_gpu, 
                model_data.HP_num_layer,
            )

        # prediction
        tag_seq = model(
            gaz_list, 
            reverse_gaz_list, 
            batch_char, 
            batch_charlen, 
            mask,
        )

        # make understandable label
        pred_label, _ = recover_label(
            tag_seq, 
            batch_label, 
            mask, 
            model_data.label_alphabet, 
            batch_charrecover,
        )

        # extend result
        ner_predictions += pred_label
        
    return ner_predictions

def get_ner_labelled_corpus(corpus, texts, ner_predictions):
    '''Pack text and entity label together into a `Dict` that has the format
        of the raw data received from CTIL. Only needed labels are preserved. 
        For the full list of entity labels, refer to the top of this file.

    Keyword arguments:
    corpus -- `Dict`. Formatted as CTIL provided.
    texts -- as produced by `get_compatible_input(...)`
    ner_predictions -- as produced by `predict(...)`
    
    Outputs:
    labelled_corpus -- `Dict`.
    '''

    labelled_corpus = []
    
    # counter to display how many NER we get
    cnt = {x: 0 for x in ['NAME', 'ORG']}

    char_label_pairs = map(lambda x, y: list(zip(x, y)), 
                           map(lambda x: x[0], texts), ner_predictions
                          )
    
    def build_ner(label_, text, len_full_text):
        return dict(
            text=text, 
            label_=label, 
            start=len_full_text,
            end=len_full_text + len(text)
        )

    for entry in corpus:
        stock_code = entry['stock_code']
        text = []

        for _ in entry['text']:
            ner = []
            full_text = ''

            for label, token in itertools.groupby(
                next(char_label_pairs), 
                key=lambda t: t[1].split('-')[-1]
            ):
                # concatenate list of character to a token
                token = ''.join(map(lambda x: x[0], token))
                
                if label == 'NAME':
                    if token in {'職業生涯', '外部職'}:
                        continue
                    ner.append(build_ner('PERSON', token, len(full_text)))
                    cnt[label] += 1
                    
                elif label == 'ORG':
                    ner.append(build_ner('ORG', token, len(full_text)))
                    cnt[label] += 1
                    
                # concatenate tokens to a doc
                full_text = full_text + token

            text.append(dict(text=full_text, ner=ner))

        labelled_corpus.append(dict(stock_code=stock_code, text=text))

    print('NER counts', cnt)
    return labelled_corpus