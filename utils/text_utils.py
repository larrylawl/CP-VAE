# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import torch
import numpy as np
from tqdm import tqdm
from random import sample

from collections import defaultdict, OrderedDict
### Ours ###

def get_preprocessor(dataset_name):
    if "gyafc" in dataset_name:
        return GYAFCPreprocessor
    else:
        raise NotImplementedError

class Preprocessor:
    def __init__(self, data_dir, subset=False, subset_count=10000):
        self.data_dir = data_dir
        self.subset = subset
        self.subset_count = subset_count
    
    def load_features(self):
        raise NotImplementedError("Implement in child class!")

class GYAFCPreprocessor(Preprocessor):
    def __init__(self, data_dir, subset=False, subset_count=10000):
        super(GYAFCPreprocessor, self).__init__(data_dir, subset=subset, subset_count=subset_count)
        self.train_fp = os.path.join(self.data_dir, "train_data.txt")
        self.dev_fp = os.path.join(self.data_dir, "dev_data.txt")
        self.test_fp = os.path.join(self.data_dir, "test_data.txt")

    def load_datasets(self):
        datasets = []

        for fp in [self.train_fp, self.dev_fp, self.test_fp]:
            dataset_sents = []
            dataset_task_labels = []

            with open(fp, "r", encoding="utf-8") as inf:
                for line in tqdm(inf):
                    label, sent = line.split("\t", 1)
                    label = int(label)
                    dataset_sents.append(sent)
                    dataset_task_labels.append(label)
            
            if self.subset and self.subset_count < len(dataset_task_labels):
                idxs = set(sample(range(0, len(dataset_task_labels)), self.subset_count))
                dataset_task_labels = [x for i, x in enumerate(dataset_task_labels) if i in idxs]
                dataset_sents = [x for i, x in enumerate(dataset_sents) if i in idxs] 
            
            datasets.append((dataset_sents, dataset_task_labels))
        return datasets

    def write_datasets(self, datasets, append_name="new"):
        assert len(datasets) == 3
        for fp, dataset in zip([self.train_fp, self.dev_fp, self.test_fp], datasets):
            root, ext = os.path.splitext(fp)
            op_fp = f"{root}_{append_name}{ext}" 
            with open(op_fp, 'w') as f:
                sents, labels = dataset
                for sent, label in zip(sents, labels):
                    f.write(f"{label}\t{sent}\n")

    def load_features(self, enc_tokenizer, dec_tokenizer, overwrite_cache):
        output = []

        for i, fp in enumerate([self.train_fp, self.dev_fp, self.test_fp]):
            if not fp: 
                output.append(([], []))
                continue
            root = os.path.splitext(fp)[0]
            cache_fn = f"{root}.pkl" if not self.subset else f"{root}_subset.pkl"
            if os.path.isfile(cache_fn) and not overwrite_cache:
                print(f"Loading cached encodings and labels: {fp}")
                output.append(torch.load(cache_fn))
            else: # create from scratch
                print(f"Creating encodings and labels: {fp}")
                if not hasattr(self, "datasets"):
                    self.datasets = self.load_datasets()

                dataset_sents, dataset_task_labels = self.datasets[i]
                enc_encodings = enc_tokenizer(dataset_sents, truncation=True, padding=True, return_tensors="pt")
                dec_encodings = dec_tokenizer(dataset_sents, truncation=True, padding=True, return_tensors="pt")
                dataset_task_labels = torch.tensor(dataset_task_labels)
                # ensures correct reconstruction loss for GPT with padding
                # read more here: https://github.com/huggingface/transformers/issues/2630
                # CEL's ignore index is default -100
                dataset_rec_labels = torch.where(dec_encodings["attention_mask"] == 1, dec_encodings["input_ids"], -100)

                cached = (enc_encodings, dec_encodings, dataset_task_labels, dataset_rec_labels, dataset_sents)
                torch.save(cached, cache_fn)
                output.append(cached)

        return output


### Original code ###

class VocabEntry(object):
    def __init__(self, vocab_size=100000):
        super(VocabEntry, self).__init__()
        self.vocab_size = vocab_size

        self.word2id = OrderedDict()
        self.unk_id = 3
        self.word2id['<pad>'] = 0
        self.word2id['<s>'] = 1
        self.word2id['</s>'] = 2
        self.word2id['<unk>'] = self.unk_id
        self.id2word_ = list(self.word2id.keys())

    def create_glove_embed(self, glove_file="data/glove.840B.300d.txt"):
        self.glove_embed = np.random.randn(len(self) - 4, 300)
        with open(glove_file) as f:
            for line in f:
                word, vec = line.split(' ', 1)

                wid = self[word]
                if wid > self.unk_id:
                    v = np.fromstring(vec, sep=" ", dtype=np.float32)
                    self.glove_embed[wid - 4, :] = v

        _mu = self.glove_embed.mean()
        _std = self.glove_embed.std()
        self.glove_embed = np.vstack([np.random.randn(4, self.glove_embed.shape[1]) * _std + _mu,
                                      self.glove_embed])

    def __getitem__(self, word):
        idx = self.word2id.get(word, self.unk_id)
        return idx if idx < self.vocab_size else self.unk_id

    def __contains__(self, word):
        return word in self.word2id

    def __len__(self):
        return min(len(self.word2id), self.vocab_size)

    def id2word(self, wid):
        return self.id2word_[wid]

    def decode_sentence(self, sentence, tensor=True):
        decoded_sentence = []
        for wid_t in sentence:
            wid = wid_t.item() if tensor else wid_t
            decoded_sentence.append(self.id2word_[wid])
        return decoded_sentence

    def build(self, sents):
        wordcount = defaultdict(int)
        for sent in sents:
            for w in sent:
                wordcount[w] += 1
        sorted_words = sorted(wordcount, key=wordcount.get, reverse=True)

        for idx, word in enumerate(sorted_words):
            self.word2id[word] = idx + 4
        self.id2word_ = list(self.word2id.keys())

class MonoTextData(object):
    def __init__(self, fname, label=False, max_length=None, vocab=None, glove=False):
        super(MonoTextData, self).__init__()
        self.data, self.vocab, self.dropped, self.labels = self._read_corpus(
            fname, label, max_length, vocab, glove)

    def __len__(self):
        return len(self.data)

    def _read_corpus(self, fname, label, max_length, vocab, glove):
        data = []
        labels = [] if label else None
        dropped = 0

        sents = []
        with open(fname) as fin:
            for line in fin:
                if label:
                    split_line = line.strip().split('\t')
                    lb = split_line[0]
                    split_line = split_line[1].split()
                else:
                    split_line = line.strip().split()

                if len(split_line) < 1:
                    dropped += 1
                    continue

                if max_length:
                    if len(split_line) > max_length:
                        dropped += 1
                        continue

                if label:
                    labels.append(int(lb))
                sents.append(split_line)
                data.append(split_line)

        if isinstance(vocab, int):
            vocab = VocabEntry(vocab)
            vocab.build(sents)
            if glove:
                vocab.create_glove_embed()
        elif vocab is None:
            vocab = VocabEntry()
            vocab.build(sents)
            if glove:
                vocab.create_glove_embed()

        # debug_data = [" ".join(x) for x in data]
        # print(debug_data[:10])
        data = [[vocab[word] for word in x] for x in data]
        # print(data[:10])

        return data, vocab, dropped, labels

    def _to_tensor(self, batch_data, batch_first, device, min_len=0):
        batch_data = [sent + [self.vocab['</s>']] for sent in batch_data]
        sents_len = [len(sent) for sent in batch_data]
        max_len = max(sents_len)
        max_len = max(min_len, max_len)
        batch_size = len(sents_len)
        sents_new = []
        sents_new.append([self.vocab['<s>']] * batch_size)
        for i in range(max_len):
            sents_new.append([sent[i] if len(sent) > i else self.vocab['<pad>']
                              for sent in batch_data])
        sents_ts = torch.tensor(sents_new, dtype=torch.long,
                                requires_grad=False, device=device)

        if batch_first:
            sents_ts = sents_ts.permute(1, 0).contiguous()

        return sents_ts, [length + 1 for length in sents_len]

    def data_iter(self, batch_size, device, batch_first=False, shuffle=True):
        index_arr = np.arange(len(self.data))
        if shuffle:
            np.random.shuffle(index_arr)
        batch_num = int(np.ceil(len(index_arr)) / float(batch_size))
        for i in range(batch_num):
            batch_ids = index_arr[i * batch_size: (i + 1) * batch_size]
            batch_data = [self.data[index] for index in batch_ids]
            batch_data, sents_len = self._to_tensor(batch_data, batch_first, device)
            yield batch_data, sents_len

    def create_data_batch_labels(self, batch_size, device, batch_first=False, min_len=5):
        sents_len = np.array([len(sent) for sent in self.data])
        sort_idx = np.argsort(sents_len)
        sort_len = sents_len[sort_idx]

        change_loc = []
        for i in range(1, len(sort_len)):
            if sort_len[i] != sort_len[i - 1]:
                change_loc.append(i)
        change_loc.append(len(sort_len))

        batch_data_list = []
        batch_label_list = []
        total = 0
        cur = 0
        for idx in change_loc:
            while cur < idx:
                batch_data = []
                batch_label = []
                nxt = min(cur + batch_size, idx)
                for id_ in range(cur, nxt):
                    batch_data.append(self.data[sort_idx[id_]])
                    batch_label.append(self.labels[sort_idx[id_]])
                cur = nxt
                batch_data, sents_len = self._to_tensor(batch_data, batch_first, device, min_len)
                batch_data_list.append(batch_data)
                batch_label_list.append(batch_label)

                total += batch_data.size(0)
                assert sents_len == ([sents_len[0]] * len(sents_len))

        return batch_data_list, batch_label_list

    def create_data_batch(self, batch_size, device, batch_first=False):
        sents_len = np.array([len(sent) for sent in self.data])
        sort_idx = np.argsort(sents_len)
        sort_len = sents_len[sort_idx]

        change_loc = []
        for i in range(1, len(sort_len)):
            if sort_len[i] != sort_len[i - 1]:
                change_loc.append(i)
        change_loc.append(len(sort_len))

        batch_data_list = []
        total = 0
        cur = 0
        for idx in change_loc:
            while cur < idx:
                batch_data = []
                nxt = min(cur + batch_size, idx)
                for id_ in range(cur, nxt):
                    batch_data.append(self.data[sort_idx[id_]])
                cur = nxt
                batch_data, sents_len = self._to_tensor(batch_data, batch_first, device)
                batch_data_list.append(batch_data)

                total += batch_data.size(0)
                assert sents_len == ([sents_len[0]] * len(sents_len))

        return batch_data_list

    def create_data_batch_feats(self, batch_size, feats, device, batch_first=False):
        sents_len = np.array([len(sent) for sent in self.data])
        print("Maximum length: %d" % max(sents_len))
        sort_idx = np.argsort(sents_len)
        sort_len = sents_len[sort_idx]

        change_loc = []
        for i in range(1, len(sort_len)):
            if sort_len[i] != sort_len[i - 1]:
                change_loc.append(i)
        change_loc.append(len(sort_len))

        batch_data_list = []
        batch_feat_list = []
        total = 0
        cur = 0
        for idx in change_loc:
            while cur < idx:
                batch_data = []
                batch_feat = []
                nxt = min(cur + batch_size, idx)
                for id_ in range(cur, nxt):
                    batch_data.append(self.data[sort_idx[id_]])
                    batch_feat.append(feats[sort_idx[id_]])
                cur = nxt
                batch_data, sents_len = self._to_tensor(batch_data, batch_first, device)
                batch_data_list.append(batch_data)
                # batch_feat = torch.tensor(
                #     batch_feat, dtype=torch.float, requires_grad=False, device=device)
                
                batch_feat = torch.tensor(
                    np.array(batch_feat), dtype=torch.float, requires_grad=False, device=device)
                batch_feat_list.append(batch_feat)

                total += batch_data.size(0)
                assert sents_len == ([sents_len[0]] * len(sents_len))

        return batch_data_list, batch_feat_list

    def data_sample(self, nsample, device, batch_first=False, shuffle=True):
        index_arr = np.arange(len(self.data))
        if shuffle:
            np.random.shuffle(index_arr)
        batch_ids = index_arr[:nsample]
        batch_data = [self.data[index] for index in batch_ids]

        batch_data, sents_len = self._to_tensor(batch_data, batch_first, device)

        return batch_data, sents_len
