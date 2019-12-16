########data augumentation ###########

import re
import string
from collections import Counter

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_tokens(s):
        if not s: return []
        return normalize_answer(s).split()
def normalize_answer(s):
    """Lower text and remove punctuation, storys and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def find_best_f1_span(target,span):
    best_f1=0
    best_i=0
    best_len=0
    best_pred=""

    tokens = span.split()
    #print(tokens)
    for i in range(len(tokens)):
        sub_tokens=tokens[i:]
        #print("i=",i)
        j=0
        for j in range(len(sub_tokens)):
            #print("j=",j)
            new_pred = ' '.join(sub_tokens[:j+1])
            #print(new_pred)
            f1=compute_f1(target,new_pred)
            #print(f1)
            if f1>best_f1:
                best_f1=f1
                best_i=i
                best_len=j+1
                best_pred=new_pred

    return best_i,best_len,best_pred



