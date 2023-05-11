import argparse


def load_normalize_bm25_scores(q_pgqs_bm25_file):
    q_pgqs_bm25 = {}
    with open(q_pgqs_bm25_file, 'r') as f:
        for line in f:
            qid, pid, bm25_score = line.strip().split('\t')
            if qid not in q_pgqs_bm25:
                q_pgqs_bm25[qid] = {}
            if pid not in q_pgqs_bm25[qid]:
                q_pgqs_bm25[qid][pid] = float(bm25_score)
            else:
                raise KeyError

    for k in q_pgqs_bm25.keys():
        max_p = max(q_pgqs_bm25[k].values())
        min_p = min(q_pgqs_bm25[k].values())
        if max_p == min_p:
            assert len(q_pgqs_bm25[k]) == 1
            for k1 in q_pgqs_bm25[k].keys():
                q_pgqs_bm25[k][k1] = 1.
        else:
            for k1 in q_pgqs_bm25[k].keys():
                q_pgqs_bm25[k][k1] = (q_pgqs_bm25[k][k1] - min_p) / (max_p - min_p)

    return q_pgqs_bm25


def get_top_seed_docs(init_bm25_file, seed_doc_num=50):
    init_top_seed_docs = {}
    with open(init_bm25_file) as bm25:
        for line in bm25:
            q_id, _, p_id, r, _, _ = line.strip().split()
            if int(r) > seed_doc_num:
                continue
            if q_id not in init_top_seed_docs:
                init_top_seed_docs[q_id] = []
            init_top_seed_docs[q_id].append(p_id)

    return init_top_seed_docs


def load_qgq_similarities(qgq_score_file, init_top_seed_docs):
    qgq_similarities = {}
    with open(qgq_score_file) as qgq_score:
        for line in qgq_score:
            q_id, gq_id, s = line.strip().split('\t')
            p_id = gq_id.split('_')[0]
            if p_id not in init_top_seed_docs[q_id]:
                continue
            if q_id not in qgq_similarities:
                qgq_similarities[q_id] = {}
            if gq_id not in qgq_similarities[q_id]:
                qgq_similarities[q_id][gq_id] = float(s)
            else:
                raise KeyError

    return qgq_similarities


def load_neighbour_docs(neighbour_doc_file):
    neighbour_docs = {}
    with open(neighbour_doc_file) as nd_file:
        for line in nd_file:
            p_id, np_id, r, _, _ = line.strip().split('\t')
            if p_id not in neighbour_docs:
                neighbour_docs[p_id] = []
            neighbour_docs[p_id].append(np_id)

    for p_id in neighbour_docs.keys():
        if p_id not in neighbour_docs[p_id]:
            neighbour_docs[p_id].append(p_id)

    return neighbour_docs


def load_gqp_relevance(gqp_score_file, init_top_seed_docs, neighbour_docs, qgq_similarities):
    gqp_scores = {}
    with open(gqp_score_file) as gqp_score:
        for line in gqp_score:
            q_id, gq_id, p_id, s = line.strip().split('\t')
            init_p_id = gq_id.split('_')[0]
            if init_p_id not in init_top_seed_docs[q_id]:
                continue
            if p_id not in neighbour_docs[init_p_id] and p_id != init_p_id:
                continue

            if q_id not in gqp_scores:
                gqp_scores[q_id] = {}
            if init_p_id not in gqp_scores[q_id]:
                gqp_scores[q_id][init_p_id] = {}
            if p_id not in gqp_scores[q_id][init_p_id]:
                gqp_scores[q_id][init_p_id][p_id] = []

            discount_s = float(s) * qgq_similarities[q_id][gq_id]
            gqp_scores[q_id][init_p_id][p_id].append(discount_s)

    return gqp_scores


def rank_neighbors(output_recall_file, gqp_scores, q_pgqs_bm25_scores, alpha, run_name, hits_num=1000, gqp_score_pooling='max'):
    pooled_p_list = {}
    for q_id in gqp_scores.keys():
        if q_id not in pooled_p_list:
            pooled_p_list[q_id] = {}
        for init_p_id in gqp_scores[q_id].keys():
            if init_p_id not in pooled_p_list[q_id]:
                pooled_p_list[q_id][init_p_id] = []

            assert len(gqp_scores[q_id][init_p_id]) == 1000 or len(gqp_scores[q_id][init_p_id]) == 1001
            for p_id in gqp_scores[q_id][init_p_id].keys():
                p_scores = gqp_scores[q_id][init_p_id][p_id]
                assert len(p_scores) == 5
                if gqp_score_pooling == 'mean':
                    p_score = sum(p_scores) / 5
                elif gqp_score_pooling == 'max':
                    p_score = max(p_scores)
                else:
                    raise NotImplementedError
                pooled_p_list[q_id][init_p_id].append((p_score, p_id))

    recall_p_list = {}
    for q_id in pooled_p_list.keys():
        if q_id not in recall_p_list:
            recall_p_list[q_id] = {}
        for init_p_id in pooled_p_list[q_id].keys():
            if init_p_id not in recall_p_list[q_id]:
                recall_p_list[q_id][init_p_id] = {}
            ranking = pooled_p_list[q_id][init_p_id]

            for rank, item in enumerate(ranking):
                score, pid = item
                recall_p_list[q_id][init_p_id][pid] = score

    with open(output_recall_file, 'w') as output_recall:
        for q_id in recall_p_list.keys():
            recall_p = {}
            for init_p_id in recall_p_list[q_id].keys():
                for p_id in recall_p_list[q_id][init_p_id].keys():
                    if p_id not in recall_p:
                        recall_p[p_id] = []
                    recall_p[p_id].append(recall_p_list[q_id][init_p_id][p_id])

            p_list = []
            for p_id in recall_p.keys():
                p_s = max(recall_p[p_id])
                p_bm25 = q_pgqs_bm25_scores[q_id][p_id]
                p_s_bm25 = alpha * p_s + (1. - alpha) * p_bm25
                p_list.append((p_s_bm25, p_id))

            if 'D' in p_id:
                recall_d = {}
                for p_s, p_id in p_list:
                    d_id = p_id.split('#')[0]
                    if d_id not in recall_d:
                        recall_d[d_id] = []
                    recall_d[d_id].append(p_s)

                d_list = []
                for d_id in recall_d.keys():
                    d_s = max(recall_d[d_id])
                    d_list.append((d_s, d_id))

                run_name += '_maxp'
                sorted_list = sorted(d_list, reverse=True)
            else:
                sorted_list = sorted(p_list, reverse=True)

            for rank, item in enumerate(sorted_list):
                if (rank + 1) > hits_num:
                    continue
                score, pid = item
                out_str = "{0}\tQ0\t{1}\t{2}\t{3}\t{4}\n".format(q_id, pid, rank + 1, score, run_name)
                output_recall.write(out_str)


def main():
    parser = argparse.ArgumentParser(description='Rank candidate neighbour docs.')
    parser.add_argument("--init_bm25_file", type=str, required=True, help='TREC format')
    parser.add_argument("--seed_doc_num", type=int, default=50, help='The num of seed doc from init BM25 run.')
    parser.add_argument("--neighbour_doc_file", type=str, required=True, help='p_id \t p_id \t rank \t recall_freq \t bm25_score')
    parser.add_argument("--qgq_score_file", type=str, required=True, help='query_id \t pseudo-query_id \t score')
    parser.add_argument("--gqp_score_file", type=str, required=True, help='query_id \t pseudo-query_id \t neighbour_id \t score')
    parser.add_argument("--q_pgqs_bm25_file", type=str, required=True, help='query_id \t p_id \t bm25_score')
    parser.add_argument("--alpha_weight", type=float, default=0.9, help='weight on BERT score')
    parser.add_argument("--output_rank_file", type=str, required=True, help='The output run file, TREC format')
    parser.add_argument("--hits_num", type=int, default=1000, help='The num of doc ranked for each query.')

    args = parser.parse_args()

    run_name = f'sd-{args.seed_doc_num}_alpha-{args.alpha_weight}'

    init_top_seed_docs = get_top_seed_docs(init_bm25_file=args.init_bm25_file, seed_doc_num=args.seed_doc_num)
    neighbour_docs = load_neighbour_docs(neighbour_doc_file=args.neighbour_doc_file)
    qgq_similarities = load_qgq_similarities(qgq_score_file=args.qgq_score_file, init_top_seed_docs=init_top_seed_docs)
    gqp_relevances = load_gqp_relevance(gqp_score_file=args.gqp_score_file, init_top_seed_docs=init_top_seed_docs,
                                        neighbour_docs=neighbour_docs, qgq_similarities=qgq_similarities)
    q_pgqs_bm25_scores = load_normalize_bm25_scores(q_pgqs_bm25_file=args.q_pgqs_bm25_file)

    rank_neighbors(output_recall_file=args.output_rank_file, gqp_scores=gqp_relevances,
                   q_pgqs_bm25_scores=q_pgqs_bm25_scores, alpha=args.alpha_weight, run_name=run_name, hits_num=args.hits_num)

    print('Done!')

if __name__ == "__main__":
    main()
