
import numpy as np
from helper_scripts import  shuffle
#import cPickle as pickle
import _pickle as pickle
import regex as re
# from util import load_dict

# import data_utils


class CLIDecodeProcessor:
    def __init__(self, source_dict, target_dict, maxlen_enc=100):

        self.source_word2id = self._load_dict(source_dict, flag='source')
        self.target_id2word = self._load_dict(target_dict, flag='target')

        self.maxlen_enc = maxlen_enc

        self._WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
        self.PAD_ID = 0
        self.EOS_ID = 1
        self.UNK_ID = 2
        self.GO_ID = 3


    def _load_dict(self, file_name, flag):

        if flag == 'source':
            _, word_dict, _ = pickle.load(open(file_name, 'r'))
        elif flag == 'target':
            _, _, word_dict = pickle.load(open(file_name, 'r'))

        return word_dict

    def word2seq(self, txt, tokenizer=None):

        word_tokens = tokenizer(txt) if tokenizer else self._basic_tokenizer(txt)
        id_tokens = [self.source_word2id.get(w, self.UNK_ID) for w in word_tokens]

        return id_tokens


    def _basic_tokenizer(self, sentence):
        words = []
        for space_separated_fragment in sentence.strip().split():
            words.extend(self._WORD_SPLIT.split(space_separated_fragment))
        return [w for w in words if w]


    def get_cli_decode_data(self, vis_input):
        vis_input = vis_input.strip()

        seq_cmd_line = []

        id_tokens_cmd_line = self.word2seq(vis_input)
        num_words = len(id_tokens_cmd_line)
        len_cmd_line = [num_words]
        if num_words > self.maxlen_enc:
            id_tokens_cmd_line = id_tokens_cmd_line[:self.maxlen_enc]
            len_cmd_line = [self.maxlen_enc]

        seq_cmd_line.append(id_tokens_cmd_line)

        return np.array(seq_cmd_line), np.array(len_cmd_line)

    def seq2words(self, seq):
        words = []
        for idx in seq:
            if idx == self.EOS_ID:
                break
            if idx in self.target_id2word:
                words.append(self.target_id2word[idx])
            else:
                words.append('UNK_ID')

        return ' '.join(words)


#Train Iterator has been made so that we can process the text in batches.
class TrainIterator:
    """Simple Bitext iterator."""
    def __init__(self, source,
                 source_dict, target_dict,
                 batch_size=128,
                 maxlen_enc=100,
                 maxlen_dec=100,
                 skip_empty=True,
                 shuffle_each_epoch=False,
                 delimiter=' +++$+++ ',
                 # sort_by_length=True,
                 maxibatch_size=20):
        
        #If you want to shuffle at each epoch then use this.(Works only in Python2.7)
        if shuffle_each_epoch:
            self.source_orig = source
            self.source = shuffle.main([self.source_orig], temporary=True)
        else:
            self.source = open(source, 'r')

        #Lets intialise our source and target dictinary.
        self.source_dict = self._load_dict(source_dict)
        self.target_dict = self._load_dict(target_dict)

        self.batch_size = batch_size
        self.maxlen_enc = maxlen_enc
        self.maxlen_dec = maxlen_dec
        self.skip_empty = skip_empty

        self._WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
        self.PAD_ID = 0
        self.EOS_ID = 1
        self.UNK_ID = 2
        self.GO_ID = 3
        self.delimiter = delimiter
        self.source_buffer = []
        self.k = batch_size

        self.end_of_data = False

    def _load_dict(self, file_name):
        _, word2id_dict, _ = pickle.load(open(file_name, 'rb'))
        return word2id_dict

    def _basic_tokenizer(self, sentence):
        words = []
        for space_separated_fragment in sentence.strip().split():
            words.extend(self._WORD_SPLIT.split(space_separated_fragment.encode()))
        return [w for w in words if w]

    def word2seq(self, txt, flag='source', tokenizer=None):
        word_tokens = tokenizer(txt) if tokenizer else self._basic_tokenizer(txt)
        if flag == 'source':
            id_tokens = [self.source_dict.get(w, self.UNK_ID) for w in word_tokens]
        elif flag == 'target':
            id_tokens = [self.target_dict.get(w, self.UNK_ID) for w in word_tokens]

        return id_tokens
    
    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])

    def reset(self):
        # if self.shuffle:
        #     self.source = shuffle.main([self.source_orig], temporary=True)
        # else:
        #     self.source.seek(0)

        self.source.seek(0)

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                line = self.source.readline()
                if line == "":
                    break
                self.source_buffer.append(line)

            self.source_buffer.reverse()

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    line = self.source_buffer.pop()
                except IndexError:
                    break

                line = line.strip().split(self.delimiter)

                ss = str(line[2])
                tt = str(line[3])

                ss_word_index = self.word2seq(ss, flag='source')
                tt_word_index = self.word2seq(tt, flag='target')

                if self.skip_empty and (not ss):
                    continue

                source.append(ss_word_index)
                target.append(tt_word_index)

                #An Extra Check to ensure source length equal to batch_size
                if len(source) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True


        source_lengths = np.array([len(s) if len(s)<self.maxlen_enc else self.maxlen_enc for s in source])
        target_lengths = np.array([len(s) if len(s)<self.maxlen_dec else self.maxlen_dec for s in target])

        source_max_len = np.max(source_lengths)
        target_max_len = np.max(target_lengths)

        batch_size = len(source)

        source_padded = np.array([(lst + [self.PAD_ID] * (source_max_len - len(lst))) if len(lst)<source_max_len else lst[:source_max_len] for lst in source])
        decoder_inputs = np.array([([self.GO_ID] + lst + [self.PAD_ID] * (target_max_len - len(lst))) if len(lst)<=target_max_len else [self.GO_ID] + lst[:target_max_len] for lst in target])
        decoder_targets = np.array([(lst + [self.EOS_ID] + [self.PAD_ID] * (target_max_len - len(lst))) if len(lst)<=target_max_len else lst[:target_max_len] + [self.EOS_ID] for lst in target])

        target_lengths += 1

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0:
            source_padded, source_lengths, decoder_inputs, target_lengths, decoder_targets = next(self)

        return source_padded, source_lengths, decoder_inputs, target_lengths, decoder_targets
