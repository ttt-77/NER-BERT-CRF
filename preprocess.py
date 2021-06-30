import os
import torch
import numpy as np
from torch.utils import data
from pytorch_pretrained_bert.tokenization import BertTokenizer
class InputExample(object):
    """A single training/test example for NER."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example(a sentence or a pair of sentences).
          words: list of words of sentence
          labels_a/labels_b: (Optional) string. The label seqence of the text_a/text_b. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        # list of words of the sentence,example: [EU, rejects, German, call, to, boycott, British, lamb .]
        self.words = words
        # list of label sequence of the sentence,like: [B-ORG, O, B-MISC, O, O, O, B-MISC, O, O]
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data.
    result of convert_examples_to_features(InputExample)
    """

    def __init__(self, input_ids, input_mask, segment_ids,  predict_mask, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.predict_mask = predict_mask
        self.label_ids = label_ids

#des: read data from file, convert to list
#input: string, innput_file - file name
#output: list, out_list - a list contains the information of each sentence (word_list, pos_tag_list, bio_pos_tag_list, ner_label_list)
class DataProcessor(object):
    @classmethod
    def _read_data(cls, input_file):
        with open(input_file) as f:
            out_list = []
            entries = f.read().strip().split("\n\n")
            for entry in entries:
                word_list = []
                pos_tag_list = []
                bio_pos_tag_list = []
                ner_label_list = []
                for line in entry.splitlines():
                    pieces = line.strip().split()
                    if len(pieces) < 1:
                        continue
                    word_list.append(pieces[0])
                    pos_tag_list.append(pieces[1])
                    bio_pos_tag_list.append(pieces[2])
                    ner_label_list.append(pieces[3])
                out_list.append([word_list, pos_tag_list, bio_pos_tag_list, ner_label_list])
        return out_list


class CoNLLDataProcessor(DataProcessor):
    '''
    CoNLL-2003
    '''

    def __init__(self):
        self._label_types = [ 'X', '[CLS]', '[SEP]', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC', 'B-ORG']
        self._num_labels = len(self._label_types)
        self._label_map = {label: i for i, label in enumerate(self._label_types)}
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "train.txt")))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "valid.txt")))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "test.txt")))

    def get_labels(self):
        return self._label_types

    def get_num_labels(self):
        return self.get_num_labels

    def get_label_map(self):
        return self._label_map

    def get_start_label_id(self):
        return self._label_map['[CLS]']

    def get_stop_label_id(self):
        return self._label_map['[SEP]']

    def _create_examples(self, all_lists):
        examples = []
        for (i, one_lists) in enumerate(all_lists):
            guid = i
            words = one_lists[0]
            labels = one_lists[-1]
            examples.append(InputExample(
                guid=guid, words=words, labels=labels))
        return examples


def example2feature(example, tokenizer, label_map, max_seq_length):

    add_label = 'X'
    # tokenize_count = []
    tokens = ['[CLS]']
    predict_mask = [0]
    label_ids = [label_map['[CLS]']]
    for i, w in enumerate(example.words):
        # use bertTokenizer to split words
        # 1996-08-22 => 1996 - 08 - 22
        # sheepmeat => sheep ##me ##at
        sub_words = tokenizer.tokenize(w)
        if not sub_words:
            sub_words = ['[UNK]']
        # tokenize_count.append(len(sub_words))
        tokens.extend(sub_words)
        for j in range(len(sub_words)):
            if j == 0:
                predict_mask.append(1)
                label_ids.append(label_map[example.labels[i]])
            else:
                # '##xxx' -> 'X' (see bert paper)
                predict_mask.append(0)
                label_ids.append(label_map[add_label])

    # truncate
    if len(tokens) > max_seq_length - 1:
        print('Example No.{} is too long, length is {}, truncated to {}!'.format(example.guid, len(tokens), max_seq_length))
        tokens = tokens[0:(max_seq_length - 1)]
        predict_mask = predict_mask[0:(max_seq_length - 1)]
        label_ids = label_ids[0:(max_seq_length - 1)]
    tokens.append('[SEP]')
    predict_mask.append(0)
    label_ids.append(label_map['[SEP]'])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    feat=InputFeatures(
                # guid=example.guid,
                # tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                predict_mask=predict_mask,
                label_ids=label_ids)

    return feat

#Creat dataset in standard form
class NerDataset(data.Dataset):
    def __init__(self, examples, tokenizer, label_map, max_seq_length):
        self.examples=examples
        self.tokenizer=tokenizer
        self.label_map=label_map
        self.max_seq_length=max_seq_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feat=example2feature(self.examples[idx], self.tokenizer, self.label_map, self.max_seq_length)
        return feat.input_ids, feat.input_mask, feat.segment_ids, feat.predict_mask, feat.label_ids

    @classmethod
    def pad(cls, batch):
        #add padding mask
        seqlen_list = [len(sample[0]) for sample in batch]
        maxlen = np.array(seqlen_list).max()

        f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: X for padding
        input_ids_list = torch.LongTensor(f(0, maxlen))
        input_mask_list = torch.LongTensor(f(1, maxlen))
        segment_ids_list = torch.LongTensor(f(2, maxlen))
        predict_mask_list = torch.ByteTensor(f(3, maxlen))
        label_ids_list = torch.LongTensor(f(4, maxlen))

        return input_ids_list, input_mask_list, segment_ids_list, predict_mask_list, label_ids_list


def dataLoader(bert_model_scale, do_lower_case, data_dir, max_seq_length, batch_size):
    conllProcessor = CoNLLDataProcessor()
    label_list = conllProcessor.get_labels()
    label_list_len = len(label_list)

    label_map = conllProcessor.get_label_map()
    train_examples = conllProcessor.get_train_examples(data_dir)
    train_examples_len = len(train_examples)
    dev_examples = conllProcessor.get_dev_examples(data_dir)
    test_examples = conllProcessor.get_test_examples(data_dir)
    tokenizer = BertTokenizer.from_pretrained(bert_model_scale, do_lower_case=do_lower_case)

    train_dataset = NerDataset(train_examples,tokenizer,label_map,max_seq_length)
    dev_dataset = NerDataset(dev_examples,tokenizer,label_map,max_seq_length)
    test_dataset = NerDataset(test_examples,tokenizer,label_map,max_seq_length)

    train_dataloader = data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=4,
                                    collate_fn=NerDataset.pad)

    dev_dataloader = data.DataLoader(dataset=dev_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=NerDataset.pad)

    test_dataloader = data.DataLoader(dataset=test_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=NerDataset.pad)
    
    start_label_id = conllProcessor.get_start_label_id()
    stop_label_id = conllProcessor.get_stop_label_id()
    return start_label_id, stop_label_id, train_dataloader, dev_dataloader, test_dataloader, label_list_len, train_examples_len
    
