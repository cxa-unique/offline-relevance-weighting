import argparse


def read_map_dict(gq_id_bm25_id_maps_file):
    maps_dict = {}
    passage_list = []
    with open(gq_id_bm25_id_maps_file) as maps:
        for line in maps:
            bm25_id, gq_id = line.strip().split()
            if bm25_id not in maps_dict:
                maps_dict[bm25_id] = gq_id
            else:
                raise KeyError
            p_id = gq_id.split('_')[0]
            if p_id not in passage_list:
                passage_list.append(p_id)

    return maps_dict, passage_list


def read_retrieval_file(gq_bm25_retrieval_file, maps_dict):
    gq_bm25_dict = {}
    with open(gq_bm25_retrieval_file) as retrieval:
        for line in retrieval:
            bm25_gq_id, _, p_id, _, s, _ = line.strip().split()
            gq_id = maps_dict[bm25_gq_id]
            if gq_id not in gq_bm25_dict:
                gq_bm25_dict[gq_id] = {}
            if p_id not in gq_bm25_dict[gq_id]:
                gq_bm25_dict[gq_id][p_id] = float(s)
            else:
                raise KeyError
    return gq_bm25_dict


def select_neighbours(passage_list, gq_bm25_dict, output_file, select_num):
    with open(output_file, 'w') as w:
        for p_id in passage_list:
            gq_bm25_ps = {}  #
            gq_ids = [p_id + '_GQ0{}'.format(i) for i in range(5)]

            for gq in gq_ids:
                if gq not in gq_bm25_dict:
                    print('Pseudo-query {} do not retrieval any document, SKIP!'.format(gq))
                    continue
                add_p_s = gq_bm25_dict[gq]

                for p in add_p_s.keys():
                    ps = add_p_s[p]
                    if p not in gq_bm25_ps:
                        gq_bm25_ps[p] = []
                    gq_bm25_ps[p].append(ps)

            np_s_list = []
            for np in gq_bm25_ps.keys():
                if len(gq_bm25_ps[np]) > 1:
                    np_s = max(gq_bm25_ps[np])  # max bm25 score
                else:
                    np_s = gq_bm25_ps[np][0]

                # (frequency of being recalled, max bm25 score, neighbour doc)
                np_s_list.append((len(gq_bm25_ps[np]), np_s, np))

            np_s_list.sort(key=lambda x: (x[0], x[1]), reverse=True)

            for rank, item in enumerate(np_s_list):
                if rank+1 > select_num:
                    continue
                recall_freq, score, pid = item
                out_str = "{0}\t{1}\t{2}\t{3}\t{4}\n".format(p_id, pid, rank+1, recall_freq, score)
                w.write(out_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Selecting neighbour documents from BM25 retrieved documents.')
    parser.add_argument('--pseudo_query_bm25_file', required=True, help='BM25 retrieval file for pseudo-queries. TREC run Format')
    parser.add_argument('--pseudo_query_map_file', required=True, help='File containing map relation.')
    parser.add_argument('--output_neighbours_file', required=True, help='Output file containing selected neighbours.')
    parser.add_argument('--select_num', default=1000, type=int, help='The number of neighbour documents.')
    args = parser.parse_args()

    maps_dict, passage_list = read_map_dict(args.pseudo_query_map_file)
    bm25_retrieval_dict = read_retrieval_file(args.pseudo_query_bm25_file, maps_dict)
    select_neighbours(passage_list, bm25_retrieval_dict, args.output_neighbours_file, args.select_num)

    print('Done!')