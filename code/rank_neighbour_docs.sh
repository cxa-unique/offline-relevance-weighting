
#init_bm25_file=./check/msmarco_passage/trec19.43.run.bm25-default.1k.txt
#seed_doc_num=50
#neighbour_doc_file=./check/msmarco_passage/trec19-43-bm25-top50_neighbor-1k.tsv
#qgq_score_file=./check/msmarco_passage/trec19-43-bm25-top50_q-gq-text.features.128.scores.tsv
#gqp_score_file=./check/msmarco_passage/trec19-43-bm25-top50_q-gq-nd-text.features.512.scores.tsv
#q_pgqs_bm25_file=./check/msmarco_passage/trec19-43-bm25-top50_q-nd-bm25-scores.tsv
#alpha_weight=0.9
#hits_num=1000
#output_rank_file=./check/msmarco_passage/trec19_passage_43_sd-${seed_doc_num}_alpha-${alpha_weight}_top1k.txt

init_bm25_file=./check/msmarco_doc/run.dl19-doc-passage.bm25-default.1k.txt
seed_doc_num=50
neighbour_doc_file=./check/msmarco_doc/dl19-43-doc-passage-bm25-top50_neighbor-1k.tsv
qgq_score_file=./check/msmarco_doc/dl19-43-doc-passage-bm25-top50_q-gq-text.features.128.scores.tsv
gqp_score_file=./check/msmarco_doc/dl19-43-doc-passage-bm25-top50_q-gq-nd-text.features.512.scores.tsv
q_pgqs_bm25_file=./check/msmarco_doc/dl19-43-doc-passage-bm25-top50_q-nd-bm25-scores.tsv
alpha_weight=0.9
hits_num=1000
output_rank_file=./check/msmarco_doc/trec19_document_43_sd-${seed_doc_num}_alpha-${alpha_weight}_top1k.txt


python rank_neighbour_docs.py --init_bm25_file ${init_bm25_file} \
                              --seed_doc_num ${seed_doc_num} \
                              --neighbour_doc_file ${neighbour_doc_file} \
                              --qgq_score_file ${qgq_score_file} \
                              --gqp_score_file ${gqp_score_file} \
                              --q_pgqs_bm25_file ${q_pgqs_bm25_file} \
                              --alpha_weight ${alpha_weight} \
                              --hits_num ${hits_num} \
                              --output_rank_file ${output_rank_file}