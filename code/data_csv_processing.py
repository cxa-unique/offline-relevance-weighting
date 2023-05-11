import logging
import torch
import numpy as np
import linecache
from torch.utils.data import DataLoader, Dataset
from tempfile import TemporaryDirectory
from pathlib import Path
from tqdm import tqdm


logger = logging.getLogger(__name__)


class PregeneratedDataset(Dataset):
    def __init__(self, data_file, max_seq_length, num_examples, cache_dir, reduce_memory=True):
        self.seq_len = max_seq_length
        self.num_samples = num_examples

        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(cache_dir)
            input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(self.num_samples, self.seq_len))
            input_masks = np.memmap(filename=self.working_dir/'input_masks.memmap',
                                    shape=(self.num_samples, self.seq_len), mode='w+', dtype=np.int32)
            segment_ids = np.memmap(filename=self.working_dir/'segment_ids.memmap',
                                    shape=(self.num_samples, self.seq_len), mode='w+', dtype=np.int32)
        else:
            raise NotImplementedError

        logging.info("Loading examples.")

        with open(data_file, 'r') as f:
            for i, line in enumerate(tqdm(f, total=self.num_samples, desc="Loading examples")):
                tokens = line.strip().split(',')
                guid = tokens[0]
                input_ids[i] = [int(id) for id in tokens[1].split()]
                input_masks[i] = [int(id) for id in tokens[2].split()]
                segment_ids[i] = [int(id) for id in tokens[3].split()]

                if i < 2:
                    logger.info("*** Example ***")
                    logger.info("guid: %s" % guid)
                    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids[i]]))
                    logger.info("input_masks: %s" % " ".join([str(x) for x in input_masks[i]]))
                    logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids[i]]))

        logging.info("Loading complete!")
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item], dtype=torch.long),
                torch.tensor(self.input_masks[item], dtype=torch.long),
                torch.tensor(self.segment_ids[item], dtype=torch.long))


def get_test_dataloader(args, data_file, sampler, batch_size):
    num_examples = int(len(linecache.getlines(data_file)))
    logger.info('Data file: {}'.format(data_file))
    logger.info('Number of examples: {}'.format(str(num_examples)))

    dataset = PregeneratedDataset(data_file, args.max_seq_length,
                                  num_examples, args.cache_file_dir)

    dataloader = DataLoader(dataset, sampler=sampler(dataset), batch_size=batch_size)

    return num_examples, dataloader