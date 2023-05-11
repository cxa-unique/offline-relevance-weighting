import argparse
import json
import os


def load_queries(query_file):
    q_texts = {}
    with open(query_file) as q_file:
        for line in q_file:
            q_id, q_text = line.strip().split('\t')
            q_texts[q_id] = q_text.strip()
    return q_texts


def load_corpus_tsv(corpus_tsv_file):
    p_texts = {}
    with open(corpus_tsv_file) as p_file:
        for line in p_file:
            p_id, p_text = line.strip().split('\t')
            if p_id not in p_texts:
                p_texts[p_id] = p_text.strip()
            else:
                raise KeyError
    return p_texts


def load_corpus_json(corpus_json_file):
    p_texts = {}
    with open(corpus_json_file) as d_file:
        for line in d_file:
            js = json.loads(line)
            p_id = js['id']
            p_text = js['contents']
            if p_id not in p_texts:
                p_texts[p_id] = p_text.strip()
            else:
                raise KeyError


def load_neighbors(neighbors_file):
    neighbors = {}
    with open(neighbors_file) as nd_file:
        for line in nd_file:
            p_id, np_id, _, _, _ = line.strip().split('\t')
            if p_id not in neighbors:
                neighbors[p_id] = []
            neighbors[p_id].append(np_id)

    for p_id in neighbors.keys():
        if p_id not in neighbors[p_id]:
            neighbors[p_id].append(p_id)
        assert len(neighbors[p_id]) == 1000 or len(neighbors[p_id]) == 1001

    return neighbors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare pseudo-query text for bm25 retrieval.')
    parser.add_argument("--init_bm25_file", type=str, required=True, help='TREC format')
    parser.add_argument('--original_query_file', required=True, help='q_id \t q_text')
    parser.add_argument('--pseudo_query_file', required=True, help='gq_id \t gq_text')
    parser.add_argument('--corpus_file', required=True, help='File containing corpus text')
    parser.add_argument("--neighbour_doc_file", type=str, required=True, help='p_id \t p_id \t rank \t recall_freq \t bm25_score')
    parser.add_argument("--output_qgq_text_file", type=str, required=True, help='q_id \t gq_id \t q_text \t gq_text')
    parser.add_argument("--output_gqp_text_file", type=str, required=True, help='q_id \t gq_id \t p_id \t gq_text \t p_text')
    parser.add_argument("--seed_doc_num", type=int, default=50, help='The num of seed doc from init BM25 run.')

    args = parser.parse_args()

    original_queries = load_queries(query_file=args.original_query_file)
    pseudo_queries = load_queries(query_file=args.pseudo_query_file)

    if os.path.splitext(os.path.basename(args.corpus_file))[1] == '.tsv':
        docs = load_corpus_tsv(corpus_tsv_file=args.corpus_file)
    elif os.path.splitext(os.path.basename(args.corpus_file))[1] == '.json':
        docs = load_corpus_json(corpus_json_file=args.corpus_file)
    else:
        raise NotImplementedError

    neighbors = load_neighbors(neighbors_file=args.neighbour_doc_file)

    with open(args.init_bm25_file) as init, \
        open(args.output_qgq_text_file, 'w') as w1, \
        open(args.output_gqp_text_file, 'w') as w2:
        for line in init:
            q_id, _, p_id, r, _, _ = line.strip().split()
            if int(r) > args.seed_doc_num:
                continue

            gq_ids = [p_id + '_GQ0{}'.format(i) for i in range(5)]

            for gq_id in gq_ids:
                w1.write(q_id + '-' + gq_id + '\t' + original_queries[q_id] + '\t' + pseudo_queries[gq_id] + '\n')

                for nd_id in neighbors[p_id]:
                    w2.write(q_id + '-' + gq_id + '-' + nd_id + '\t' + pseudo_queries[gq_id] + '\t' + docs[nd_id] + '\n')