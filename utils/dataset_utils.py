import torch
from torch.utils.data import Dataset, Subset

def get_dataset(dataset_name):
    if "gyafc" == dataset_name:
        return GYAFCDataset
    else:
        raise NotImplementedError

class GYAFCDataset(Dataset):
    "Follows: https://huggingface.co/transformers/v3.2.0/custom_datasets.html"

    def __init__(self, enc_encs, dec_encs, task_labels, rec_labels, sents):
        self.enc_encs = enc_encs
        self.dec_encs = dec_encs
        self.labels = task_labels
        self.rec_labels = rec_labels
        self.sents = sents

    def __getitem__(self, idx):
        item = {}
        for key, val in self.enc_encs.items():
            item[f"enc_{key}"] = val[idx]

        for key, val in self.dec_encs.items():
            item[f"dec_{key}"] = val[idx]

        item['labels'] = self.labels[idx]
        item["rec_labels"] = self.rec_labels[idx]
        
        # item = {
        #     "labels": self.labels[idx]
        # }

        # # looping through all sentences
        # for i in range(len(self.encodings)):
        #     enc = self.encodings[i]
        #     for k, v in enc.items(): # encodings from tokenizer: input_ids, atten_mask, etc
        #         item[f"{k}_sent_{i}"] = v[idx]
        # item['no_sentence'] = i + 1  # for meta CLS token
        return item
    
    def __len__(self):
        return len(self.labels)

    def get_subset_specified_labels(self, label, nsamples=10):
        assert label in self.labels
        idxs = []
        for i,l in enumerate(self.labels):
            if len(idxs) == nsamples:
                break
            if label == l:
                idxs.append(i)
        return Subset(self, idxs)