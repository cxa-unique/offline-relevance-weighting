import argparse


def load_pseudo_query(gq_text_file):
    gq_text_dict = {}
    with open(gq_text_file, 'r') as gq_file:
        for line in gq_file:
            gq_id, gq_text = line.strip().split('\t')
            if gq_id not in gq_text_dict:
                gq_text_dict[gq_id] = gq_text.strip()
            else:
                raise KeyError

    return gq_text_dict


def collect_used_pseudo_query(init_bm25_top_file, seed_doc_num=50):
    init_top_dict = {}
    with open(init_bm25_top_file) as init:
        for line in init:
            q_id, _, p_id, r, _, _ = line.strip().split()
            if q_id not in init_top_dict:
                init_top_dict[q_id] = []
            if int(r) > seed_doc_num:
                continue
            init_top_dict[q_id].append(p_id)

    gq_list = []
    for q_id in init_top_dict.keys():
        for p_id in init_top_dict[q_id]:
            gqs = [p_id + '_GQ0{}'.format(i) for i in range(5)]
            gq_list.extend(gqs)

    gq_list_uniq = sorted(set(gq_list))

    return gq_list_uniq


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare pseudo-query text for bm25 retrieval.')
    parser.add_argument("--init_bm25_file", type=str, required=True, help='TREC format')
    parser.add_argument('--all_pseudo_query_file', required=True, help='From [docTTTTTquery](https://github.com/castorini/docTTTTTquery)')
    parser.add_argument('--pseudo_query_map_file', required=True, help='File containing map relation: pseudo-query_id_4_bm25 \t pseudo-query_id')
    parser.add_argument('--pseudo_query_bm25_id_file', required=True, help='pseudo-query_id_4_bm25 \t pseudo-query_text')
    parser.add_argument('--pseudo_query_gq_id_file', required=True, help='pseudo-query_id \t pseudo-query_text')
    parser.add_argument("--seed_doc_num", type=int, default=50, help='The num of seed doc from init BM25 run.')

    args = parser.parse_args()

    pseudo_queries = load_pseudo_query(gq_text_file=args.all_pseudo_query_file)
    used_pseudo_queries = collect_used_pseudo_query(init_bm25_top_file=args.init_bm25_file, seed_doc_num=args.seed_doc_num)

    with open(args.pseudo_query_bm25_id_file, 'w') as gqt1, \
        open(args.pseudo_query_gq_id_file, 'w') as gqt2, \
        open(args.pseudo_query_map_file, 'w') as ids:
        for i, gq in enumerate(used_pseudo_queries):
            gq_text = pseudo_queries[gq]
            gqt1.write(str(i) + '\t' + gq_text + '\n')
            gqt2.write(gq + '\t' + gq_text + '\n')
            ids.write(str(i) + '\t' + gq + '\n')

    print('Done!')
