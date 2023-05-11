from pyserini.index import IndexReader
import argparse


def load_query(query_file):
    query_dict = {}
    with open(query_file, 'r') as q_file:
        for line in q_file:
            q_id, q_text = line.strip().split('\t')
            if q_id not in query_dict:
                query_dict[q_id] = q_text.strip()
            else:
                raise KeyError
    return query_dict


def load_uniq_neighbors_per_query(init_bm25_file, neighbors_file, seed_doc_num=50):
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

    q_neighbors = {}
    with open(init_bm25_file) as init:
        for line in init:
            q_id, _, p_id, r, _, _ = line.strip().split()
            if int(r) > seed_doc_num:
                continue
            if q_id not in q_neighbors:
                q_neighbors[q_id] = []
            q_neighbors[q_id].extend(neighbors[p_id])

    for q_id in q_neighbors.keys():
        p_neighbor_ids = sorted(set(q_neighbors[q_id]))
        q_neighbors[q_id] = p_neighbor_ids

    return q_neighbors


def get_bm25_score_from_index(index_reader, q_neighbors, query_dict, output_score_file):
    with open(output_score_file, 'w') as w:
        for q_id in q_neighbors.keys():
            for p_id in q_neighbors[q_id]:
                score = index_reader.compute_query_document_score(p_id, query_dict[q_id])
                w.write(q_id + '\t' + p_id + '\t' + str(score) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Rank candidate neighbour docs.')
    parser.add_argument("--query_file", type=str, required=True, help='Query file: q_id \t q_text')
    parser.add_argument("--init_bm25_file", type=str, required=True, help='TREC format')
    parser.add_argument("--seed_doc_num", type=int, default=50, help='The num of seed doc from init BM25 run.')
    parser.add_argument("--neighbour_doc_file", type=str, required=True, help='p_id \t p_id \t rank \t recall_freq \t bm25_score')
    parser.add_argument("--output_q_pgqs_bm25_file", type=str, required=True, help='query_id \t p_id \t bm25_score')
    parser.add_argument("--index_dir", type=str, required=True, help='The index dir from Anserini.')

    args = parser.parse_args()

    # index_reader = IndexReader('./anserini/indexes/lucene-index.msmarco-passage-t5-40.pos+docvectors+raw')
    # index_reader = IndexReader('./anserini/indexes/lucene-index.msmarco-doc-expanded-passage.pos+docvectors+raw')

    index_reader = IndexReader(args.index_dir)

    queries = load_query(query_file=args.query_file)
    q_uniq_neighbors = load_uniq_neighbors_per_query(init_bm25_file=args.init_bm25_file,
                                                     neighbors_file=args.neighbour_doc_file,
                                                     seed_doc_num=args.seed_doc_num)

    get_bm25_score_from_index(index_reader=index_reader, q_neighbors=q_uniq_neighbors,
                              query_dict=queries, output_score_file=args.output_q_pgqs_bm25_file)

    print('Done!')


if __name__ == "__main__":
    main()
