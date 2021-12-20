import logging
import os
import pickle
from multiprocessing import Pool
from typing import Tuple

import pandas as pd
import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import numpy as np
from itertools import chain
from torch.utils.data import Sampler
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def preprocess_data(data):
    input_text, target_text, encoder_tokenizer, decoder_tokenizer, args = data

    input_text = encoder_tokenizer.encode(
        input_text, max_length=args.max_seq_length, pad_to_max_length=True, return_tensors="pt",
    )

    target_text = decoder_tokenizer.encode(
        target_text, max_length=args.max_length, pad_to_max_length=True, return_tensors="pt"
    )
    return torch.flatten(input_text), torch.flatten(target_text)


class Seq2SeqDataset(Dataset):
    def __init__(self, encoder_tokenizer, decoder_tokenizer, args, data, mode):
        self.encoder_tokenizer = encoder_tokenizer
        cached_features_file = os.path.join(
            args.cache_dir, args.model_name + f"_{mode}_cached_" + str(args.max_seq_length) + str(len(data))
        )

        if os.path.exists(cached_features_file) and (
                (not args.reprocess_input_data and not args.no_cache)
                or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s", args.cache_dir)

        data = [
            (input_text, target_text, encoder_tokenizer, decoder_tokenizer, args)
            for input_text, target_text in zip(data["input_text"], data["target_text"])
        ]

        if args.use_multiprocessing:
            with Pool(args.process_count) as p:
                self.examples = list(
                    tqdm(
                        p.imap(preprocess_data, data, chunksize=args.multiprocessing_chunksize),
                        total=len(data),
                        disable=args.silent,
                    )
                )
        else:
            self.examples = [preprocess_data(d) for d in tqdm(data, disable=args.silent)]

        logger.info(" Saving features into cached file %s", cached_features_file)
        with open(cached_features_file, "wb") as handle:
            pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def get_source_seq_lens(self):
        return [torch.sum(ex[0] != self.encoder_tokenizer.pad_token_id).item() for ex in self.examples]


def preprocess_data_bart(data):
    input_text, target_text, tokenizer, args = data
    input_ids = tokenizer.batch_encode_plus(
        [input_text], max_length=args.max_seq_length, padding='longest', truncation=True, return_tensors="pt",
    )

    target_ids = tokenizer.batch_encode_plus(
        [target_text], max_length=args.max_length, padding='longest', truncation=True, return_tensors="pt"
    )

    return {
        "source_ids": input_ids["input_ids"].squeeze(),
        "source_mask": input_ids["attention_mask"].squeeze(),
        "target_ids": target_ids["input_ids"].squeeze(),
    }


class SimpleSummarizationDataset(Dataset):
    def __init__(self, tokenizer, args, data, mode):
        self.tokenizer = tokenizer

        cached_features_file = os.path.join(
            args.cache_dir, args.model_name + f"_{mode}_cached_" + str(args.max_seq_length) + str(len(data))
        )

        if os.path.exists(cached_features_file) and (
                (not args.reprocess_input_data and not args.no_cache)
                or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s", args.cache_dir)
        data = [
            (input_text, target_text, tokenizer, args)
            for input_text, target_text in zip(data["input_text"], data["target_text"])
        ]

        if args.use_multiprocessing:
            torch.multiprocessing.set_sharing_strategy('file_system')
            with Pool(args.process_count) as p:
                self.examples = list(
                    tqdm(
                        p.imap(preprocess_data_bart, data, chunksize=args.multiprocessing_chunksize),
                        total=len(data),
                        disable=args.silent,
                    )
                )
        else:
            self.examples = [preprocess_data_bart(d) for d in tqdm(data, disable=args.silent)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def get_source_seq_lens(self):
        return [torch.sum(ex['source_mask'], dim=-1).item() for ex in self.examples]


class PadCollateFn:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, pad_vals, for_bart=False, dim=0):
        """
        args:
            pad_vals - List[int], pad vals corresponding to each batch element
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        self.for_bart = for_bart
        self.pad_vals = pad_vals

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        stack_batch = dict()
        for ith, key in enumerate(batch[0].keys()):
            pad_val = self.pad_vals[ith]
            items = [item[key] for item in batch]
            max_len = max(map(len, items))
            temp_batch = [pad_tensor(x, pad=max_len, pad_val=pad_val, dim=self.dim) for x in items]
            stack_batch[key] = torch.stack(temp_batch, dim=0)
        return stack_batch

    def __call__(self, batch):
        return self.pad_collate(batch)


def pad_tensor(vec, pad, dim, pad_val=0):
    """
    args:
        vec - tensor to pad
        pad - int, the size to pad to
        dim - int, dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size).fill_(pad_val)], dim=dim).type_as(vec)


# ===================================================DataLoader====================================
class BartSeq2seqDataLoader(DataLoader):

    def __init__(self, dataset, batch_size, sampler, pad_vals, num_workers, pin_memory=True):
        super().__init__(dataset=dataset,
                         batch_size=batch_size,
                         sampler=sampler,
                         num_workers=num_workers,
                         pin_memory=pin_memory,
                         collate_fn=PadCollateFn(pad_vals, dim=0, for_bart=True))


# ===================================================Sampler====================================
class BucketSampler(Sampler):
    r"""
     `Random Sampler` with bucket. Elements of similar length can be randomly extracted
    """

    def __init__(self, seq_lens, data_source=None, batch_size=16, num_buckets=20):
        r"""

        :param int batch_size:  default is None，
        """
        super(BucketSampler, self).__init__(data_source)
        self.batch_size = batch_size
        self.num_buckets = num_buckets
        self.seq_lens = seq_lens

    def __call__(self):
        if self.batch_size is None:
            raise RuntimeError("batch_size is None.")
        total_sample_num = len(self.seq_lens)

        bucket_indexes = []
        assert total_sample_num >= self.num_buckets, "The number of samples is smaller than the number of buckets."
        num_sample_per_bucket = total_sample_num // self.num_buckets
        for i in range(self.num_buckets):
            bucket_indexes.append([num_sample_per_bucket * i, num_sample_per_bucket * (i + 1)])
        bucket_indexes[-1][1] = total_sample_num

        sorted_seq_lens = list(sorted([(idx, seq_len) for
                                       idx, seq_len in zip(range(total_sample_num), self.seq_lens)],
                                      key=lambda x: x[1]))
        batchs = []

        left_init_indexes = []
        for b_idx in range(self.num_buckets):
            start_idx = bucket_indexes[b_idx][0]
            end_idx = bucket_indexes[b_idx][1]
            sorted_bucket_seq_lens = sorted_seq_lens[start_idx:end_idx]
            left_init_indexes.extend([tup[0] for tup in sorted_bucket_seq_lens])
            num_batch_per_bucket = len(left_init_indexes) // self.batch_size
            np.random.shuffle(left_init_indexes)
            for i in range(num_batch_per_bucket):
                batchs.append(left_init_indexes[i * self.batch_size:(i + 1) * self.batch_size])
            left_init_indexes = left_init_indexes[num_batch_per_bucket * self.batch_size:]
        if left_init_indexes != 0:
            batchs.append(left_init_indexes)
        np.random.shuffle(batchs)

        return list(chain(*batchs))

    def __iter__(self):
        indices = self.__call__()
        return iter(indices)

    def __len__(self):
        return len(self.seq_lens)


class SortedSampler(Sampler):
    r"""
    Sorting according to the length of the sample is mainly used during testing,
    which can speed up the testing (because the padding is reduced)
    """

    def __init__(self, seq_lens, data_source=None, descending=True):
        """

        :type data_source: Sized
        :type seq_lens: List[int]
        :param bool descending: 是否降序排列
        """
        super(SortedSampler, self).__init__(data_source)
        self.descending = descending
        self.seq_lens = seq_lens
        self.orders = self.__call__()

    def __call__(self):
        seq_lens = np.array(self.seq_lens)
        orders = np.argsort(seq_lens).tolist()
        if self.descending:
            orders = orders[::-1]
        return orders

    def __iter__(self):
        return self.orders

    def __len__(self):
        return len(self.seq_lens)
