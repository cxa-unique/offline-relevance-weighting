import os
import logging
import argparse
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.utils.data import SequentialSampler
from bert.modeling import BertForSequenceClassification
from data_csv_processing import get_test_dataloader


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def do_eval(model, eval_dataloader, device):
    scores = []
    model.eval()
    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids = batch_

            logits = model(input_ids, segment_ids, input_mask)

            probs = F.softmax(logits, dim=1)[:, 1]
            scores.append(probs.detach().cpu().numpy())

    result = {}
    result['scores'] = np.concatenate(scores)

    return result


def save_scores(args, scores):
    ids_list = []
    with open(args.input_features_file, mode='r') as ref_file:
        for line in ref_file:
            guid = line.strip().split(',')[0]
            ids_list.append(guid.split('-'))
    assert len(scores) == len(ids_list)

    with open(args.output_score_file, 'w') as w:
        for idx, score in enumerate(scores):
            out_str = '\t'.join(ids_list[idx]) + f'\t{score}\n'
            w.write(out_str)

    print('Done!')


def main():
    parser = argparse.ArgumentParser(description='BERT ranker/scorer.')
    parser.add_argument("--device",
                        default=None,
                        type=str,
                        required=True,
                        help="The device you will run on.")
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The model dir for inference.")
    parser.add_argument("--input_features_file",
                        default=None,   
                        type=str, 
                        required=True,   
                        help="The features data file")
    parser.add_argument("--output_score_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The output score file")
    parser.add_argument("--cache_file_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The cache dir for data loader")
    parser.add_argument("--eval_batch_size",
                        default=None,
                        type=int,
                        help="Batch size for eval.")
    parser.add_argument("--max_seq_length",
                        default=None,
                        type=int,
                        help="The maximum total input sequence length after tokenization.")

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if not os.path.exists(args.cache_file_dir):
        os.makedirs(args.cache_file_dir)

    num_labels = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file_dir = os.path.join(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(model_file_dir, num_labels=num_labels)
    model.to(device)

    _, eval_dataloader = get_test_dataloader(args, args.input_features_file, SequentialSampler, args.eval_batch_size)

    res = do_eval(model, eval_dataloader, device)
    scores = res['scores']

    save_scores(args, scores)

    if os.path.exists(args.cache_file_dir):
        import shutil
        shutil.rmtree(args.cache_file_dir)


if __name__ == "__main__":
    main()