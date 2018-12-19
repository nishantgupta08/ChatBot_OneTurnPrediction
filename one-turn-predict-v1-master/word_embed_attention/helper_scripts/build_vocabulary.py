import collections
import operator
import pandas as pd
import numpy as np
import re
import pickle
import datetime
import time

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
# _START_VOCAB = [_PAD, _GO, _EOS, _UNK]
_START_VOCAB_ENC = [_PAD, _EOS, _UNK]
_START_VOCAB_DEC = [_PAD, _EOS, _UNK, _GO]

# PAD_ID = 0
# EOS_ID = 1
# UNK_ID = 2
# GO_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")


vocab_list_file = "../output/rnn_vocab_list_file.p"

def _basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]

def _compute_vocab_size_stats(sorted_vocab_counts, vocab_size):
    pass

def _build_vocab(lines, vocab_size, source, tokenizer=None):
    tokens = tokenizer(lines) if tokenizer else _basic_tokenizer(lines)

    vocab_counts = {}
    for w in tokens:
        w = str(w).strip()
        if w in vocab_counts:
            vocab_counts[w] += 1
        else:
            vocab_counts[w] = 1

    # Write word dsitributions to a file
    date_str = datetime.datetime.now().strftime("%d_%m_%Y")

    sorted_vocab_counts = sorted(list(vocab_counts.items()), key=operator.itemgetter(1), reverse=True)
    print("======================================================")
    print("Source type: ", str(source))
    print("Actual Vocab Size: ", len(sorted_vocab_counts))

    # import pdb; pdb.set_trace()

    with open("vocab_data/"+str(source)+"_vocab_count_distribution_first_response_jan_2018_p1_" + date_str +".tsv", "w") as f:
        for word, cnt in sorted_vocab_counts:
            f.write(word + ":" + str(cnt) + '\n')

    _compute_vocab_size_stats(sorted_vocab_counts, vocab_size)

    if source == 'user':
        word_order = _START_VOCAB_ENC + sorted(vocab_counts, key=vocab_counts.get, reverse=True)
    elif source == 'agent':
        word_order = _START_VOCAB_DEC + sorted(vocab_counts, key=vocab_counts.get, reverse=True)

    if vocab_size is not None:
        if len(word_order) > vocab_size:
            word_order = word_order[:vocab_size]
        else:
            print("Vocab Size is higher than the num unique words in the file ")
            print("Unique words: ", len(word_order))
            print("Vocab size: ", vocab_size)

    word_2_id_dict = dict([(x, y) for (y, x) in enumerate(word_order)])
    id_2_word_dict = dict([(y, x) for (y, x) in enumerate(word_order)])
    print("Word2ID Dict Len: ", len(word_2_id_dict))
    print("ID2Word Dict Len ", len(id_2_word_dict))

    # pickle.dump([word_order, word_2_id_dict, id_2_word_dict],open("data/dictionary_"+ date_str +"_dds_ddc_ce.p", "wb"))
    pickle.dump([word_order, word_2_id_dict, id_2_word_dict],open("vocab_data/"+str(source)+"_first_response_jan_2018_p1_"+ date_str +".p", "wb"))
    print("======================================================")

def main(file_name, col_sep=" +++$+++ ", combined_dict=True, vocab_size_enc=None, vocab_size_dec=None):

    user_txt = ""
    agent_txt = ""

    with open(file_name, 'r') as read_handler:
        for line in read_handler:
            cols = line.split(col_sep)
            user_txt = user_txt + " " + str(cols[2])
            agent_txt = agent_txt + " " + str(cols[3])

    if combined_dict:
        total_txt = user_txt + " " + agent_txt
        _build_vocab(total_txt, vocab_size)

    else:
        _build_vocab(user_txt, vocab_size_enc, source='user')
        _build_vocab(agent_txt, vocab_size_dec, source='agent')


if __name__ == "__main__":
    start_time = time.time()

    VOCAB_SIZE_ENC = None
    VOCAB_SIZE_DEC = None
    # VOCAB_SIZE_ENC = 1687  # (USER) Words with at least a frequency of 3
    # VOCAB_SIZE_DEC = 1627  # (AGENT) Words with at least a frequency of 3

    #file_name = "data/first-response-jan-2018-processed-p1.txt"
    file_name = "D:\\DeepLearningNewLearning\\chatbot\\one-turn-predict-v1-master\\one-turn-predict-v1-master\\data\\sample.txt"
    # file_name = "data/first-response-jan-2018-processed-p1-p2-train.txt"
    main(file_name, col_sep=" +++$+++ ", combined_dict=False, vocab_size_enc = VOCAB_SIZE_ENC, vocab_size_dec = VOCAB_SIZE_DEC)

    print("Script Time: ", round(time.time()-start_time, 0), ' Seconds')
