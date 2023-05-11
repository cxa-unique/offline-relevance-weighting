import bert.tokenization as tokenization
import argparse


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    def __init__(self, guid, input_ids, input_mask, segment_ids):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_examples_to_features(example, max_seq_length, tokenizer):
    query = tokenization.convert_to_unicode(example.text_a)
    query_tokens = tokenization.convert_to_bert_input(
        text=query,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        add_cls=True,
        add_sep=True,
        truncate_warning=False)

    p_content = tokenization.convert_to_unicode(example.text_b)
    passage_tokens = tokenization.convert_to_bert_input(
        text=p_content,
        max_seq_length=(max_seq_length - len(query_tokens)),
        tokenizer=tokenizer,
        add_cls=False,
        add_sep=True,
        truncate_warning=True)

    input_ids = query_tokens + passage_tokens
    segment_ids = [0] * len(query_tokens) + [1] * len(passage_tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    features = InputFeatures(guid=example.guid,
                             input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids)
    return features


def get_rank_task_features(input_text_pairs_file,  output_features_file, tokenizer, max_seq_length):
    with open(input_text_pairs_file, 'r') as text_file, \
            open(output_features_file, 'w') as csv_file:
        for i, line in enumerate(text_file):
            guid, text_1, text_2 = line.strip().split('\t')

            example = InputExample(guid=guid, text_a=text_1, text_b=text_2)
            features = convert_examples_to_features(example, max_seq_length, tokenizer)

            input_ids_str = " ".join([str(id) for id in features.input_ids])
            input_mask_str = " ".join([str(id) for id in features.input_mask])
            segment_ids_str = " ".join([str(id) for id in features.segment_ids])
            csv_file.write(guid + ',' + input_ids_str + ',' + input_mask_str + ',' + segment_ids_str + '\n')

            if i < 1:
                print("*** Example ***")
                print("guid: %s" % example.guid)
                print("input_ids: %s" % input_ids_str)
                print("input_mask: %s" % input_mask_str)
                print("segment_ids: %s" % segment_ids_str)


def main():
    parser = argparse.ArgumentParser(description='Convert text to features for BERT model.')
    parser.add_argument("--input_text_pairs_file", type=str, required=True, help='Text file: guid \t text_1 \t text_2 \n')
    parser.add_argument("--output_features_file", type=str, required=True, help='Features file: guid, input_ids_str, input_mask_str, segment_ids \n')
    parser.add_argument("--max_seq_length", type=int, default=512)

    args = parser.parse_args()

    vocab_dir = './bert/vocab.txt'
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_dir, do_lower_case=True)

    print('Starting convert to features...')
    get_rank_task_features(args.input_text_pairs_file, args.output_features_file, tokenizer, args.max_seq_length)
    print('Convert to csv done!')


if __name__ == "__main__":
    main()
