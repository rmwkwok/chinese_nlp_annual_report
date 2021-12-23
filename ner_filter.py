# Hard-coded rule for filtering potentially invalid NER
import re

def is_valid_name(text):
    '''check if `text` is a valid name.

    Keyword arguments:
    text -- `str`. Name.
    
    Outputs:
    `Bool`. 
    '''
    
    # English-only name that can carry dot
    if re.match(r'^[a-zA-Z.]*$', text):
        
        # Expecting a minimum length
        if len(text) < 6:
            return False
    
    # Chinese-only name
    elif re.match(r'^[\u4e00-\u9fff]*$', text):
        
        # Expecting a length range
        if len(text) < 2 or len(text) > 4:
            return False
        
        # Model-related common exceptions:
        if text in ['於集團擔', '附註']:
            return False
    
    # Other combination of text is not a name
    else:
        return False
    
    return True

def is_valid_org(text):
    '''check if `text` is a valid organization name.

    Keyword arguments:
    text -- `str`. Name.
    
    Outputs:
    `Bool`. 
    '''

    # English-only name that can carry dot
    if re.match(r'^[a-zA-Z.]*$', text):
        
        # Expecting a minimum length
        if len(text) < 6:
            return False
    
    # Chinese-only name
    elif re.match(r'^[\u4e00-\u9fff]*$', text):
        
        # Expecting a minimum length. XX集團 at least.
        if len(text) < 4:
            return False
    
    # Other combination of text is not a name
    else:
        return False
    
    return True

def filter_ner(labelled_corpus):
    '''Filter NER.

    Keyword arguments:
    labelled_corpus -- `Dict`
    
    Outputs:
    labelled_corpus --- `Dict`. With invalid NER removed
    valid_org --- `List`. List of valid org remained in `labelled_corpus`
    valid_name --- `List`. List of valid name remained in `labelled_corpus`
    invalid_org --- `List`. List of invalid org removed
    invalid_name --- `List`. List of invalid name removed
    '''
    
    ner_cnt = {'NAME': 0, 'ORG': 0}
    valid_org = []
    valid_person = []
    invalid_org = []
    invalid_person = []

    for stock in labelled_corpus:
        for doc in stock['text']:
            keep = []

            for ner in doc['ner']:
                text = ner['text']
                label = ner['label_']

                if label == 'NAME':
                    if is_valid_name(text):
                        keep.append(True)
                        valid_person.append(text)
                        ner_cnt['NAME'] += 1
                    else:
                        keep.append(False)
                        invalid_person.append(text)

                elif label == 'ORG':
                    if is_valid_org(text):
                        keep.append(True)
                        valid_org.append(text)
                        ner_cnt['ORG'] += 1
                    else:
                        keep.append(False)
                        invalid_org.append(text)

                else:
                    keep.append(False)

            doc['ner'] = [ner for k, ner in zip(keep, doc['ner']) if k]

    print('NER counts', ner_cnt)
    print('# Unique Valid person/org: %d/%d'%(len(set(valid_person)), len(set(valid_org))))
    print('# Unique Invalid person/org: %d/%d'%(len(set(invalid_person)), len(set(invalid_org))))
    
    return labelled_corpus, valid_org, valid_person, invalid_org, invalid_person