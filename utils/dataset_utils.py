import torch
import random
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
        self.labels_type = self.get_labels_type()
        self.label_to_idxs = self.get_label_to_idxs()

    def _get_item_helper(self, idx):
        item = {}
        for key, val in self.enc_encs.items():
            item[f"enc_{key}"] = val[idx]

        for key, val in self.dec_encs.items():
            item[f"dec_{key}"] = val[idx]

        item['labels'] = self.labels[idx]
        item["rec_labels"] = self.rec_labels[idx]

        return item

    def __getitem__(self, idx):
        item = self._get_item_helper(idx)
        label = self.labels[idx].item()
        buddy_idx = random.choice(self.label_to_idxs[label])
        buddy_item = self._get_item_helper(buddy_idx)
        assert label == self.labels[buddy_idx]
    
        return item, buddy_item
    
    def __len__(self):
        return len(self.labels)

    def get_labels_type(self):
        labels_list = [l.item() for l in self.labels]
        return set(labels_list)

    def get_subset_specified_labels(self, label, nsamples=10):
        assert label in self.labels_type
        idxs = []
        for i,l in enumerate(self.labels):
            if len(idxs) == nsamples:
                break
            if label == l:
                idxs.append(i)
        return Subset(self, idxs)

    def get_subset_specified_labels_idx(self, label):
        assert label in self.labels_type
        idxs = []
        for i,l in enumerate(self.labels):
            if label == l.item():
                idxs.append(i)
        return idxs

    def get_label_to_idxs(self):
        op = {}
        for label in self.labels_type:
            idxs = self.get_subset_specified_labels_idx(label)
            op[label] = idxs
        return op
