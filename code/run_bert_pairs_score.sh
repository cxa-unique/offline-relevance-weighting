
device=6
batch_size=50

#max_seq_length=128
#model_dir=/home1/cxa/ir_data/models/parade_bert_msmarco/bert_models_on_MSMARCO/vanilla_bert_base_on_MSMARCO
#input_features_file=./check/msmarco_passage/trec19-43-bm25-top50_q-gq-text.features.128.csv
#output_score_file=./check/msmarco_passage/trec19-43-bm25-top50_q-gq-text.features.128.scores.tsv

max_seq_length=512
model_dir=/home1/cxa/ir_data/models/parade_bert_msmarco/bert_models_on_MSMARCO/vanilla_bert_base_on_MSMARCO
input_features_file=./check/msmarco_passage/trec19-43-bm25-top50_q-gq-nd-text.features.512.csv
output_score_file=./check/msmarco_passage/trec19-43-bm25-top50_q-gq-nd-text.features.512.scores.tsv


python run_bert_pairs_score.py --device ${device} \
                               --model_dir ${model_dir} \
                               --input_features_file ${input_features_file} \
                               --output_score_file ${output_score_file} \
                               --cache_file_dir ./cache${device} \
                               --eval_batch_size ${batch_size} \
                               --max_seq_length ${max_seq_length}