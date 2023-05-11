# Contextualized Offline Relevance Weighting

In this page, we provide an instruction on how to implement our retrieval method with a limited test queries 
in a restricted environment.
We use [Anserini](https://github.com/castorini/anserini) and [Pyserini](https://github.com/castorini/pyserini/)
toolkits for the implementations of BM25 method, which is the basic component in our retrieval method. 

### 1. Offline Preparation
During offline, we first need to generate a few pseudo-queries for each document in the corpus, and then seek 
the neighbour documents for one document with the help of its pseudo-queries, and also compute the relevance 
scores between pseudo-queries and documents.

**Pseudo-Query:** We use the generated queries by T5 model released in [docTTTTTquery](https://github.com/castorini/docTTTTTquery)
repo. You can find that there are 80 predicted queries for each MS MARCO passage, and 10 predicted queries for 
each MS MARCO document. In our experiments, we only use the **first 5 predicted queries**, and save all of them
as in the following file:
```
All pseudo-query text file:
Format: pseudo-query_id \t pseudo-query_text
% 'pseudo-query_id' = 'document_id' + '_GQ0{0-4}'
```

For the operability of experiments, we can only perform the offline preparation on the seed documents from queries 
waiting for testing, rather than all documents in the corpus.
For example, if we want to evaluate 43 test queries in TREC 2019 DL set, we only collect related pseudo-queries for 
top-30/50 seed documents from BM25 results for each test queries, as seen in `prepare_pseudo_queries`. 
As Anserini does not support `str` query id for BM25 retrieval, you will find that the following two files are also 
produced:
```
1) Pseudo-Query text file: pseudo-query_id_4_bm25 \t pseudo-query_text  
2) Pseudo-query map file: pseudo-query_id_4_bm25 \t pseudo-query_id
% 'pseudo-query_id_4_bm25' is used for BM25 retrieval in Anserini, it usually starts from 0.
```
And then we can use `file 1)` to do BM25 retrieval by Anserini toolkit for the construction of neighbour documents.

**Neighbour Documents:** For each document, each of its pseudo-queries is used to recall 1,000 documents from the corpus
using BM25 (default parameter settings in Anserini), and we then select 1,000 documents from 5,000 documents according 
to the recall frequency and BM25 scores, to form the final neighbour document set.
```
% Select neighbour documents
python select_neighbour_docs.py --pseudo_query_bm25_file ${path_to_pseudo_query_bm25_file} \
                                --pseudo_query_map_file ${path_to_map_file} \
                                --output_neighbours_file ${path_to_neighbours_file}
```

**Pre-computation of Relevance:** We use BERT ranker to weigh the relevance between pseudo-query and neighbour document. 
The checkpoint of the BERT-Base used in our experiments is released in [PARADE](https://github.com/canjiali/PARADE) repo.
We need to prepare a text file that contains (5*1000) pairs between (5) pseudo-queries and (1000) neighbour documents 
for one document, like `query_id \t pseudo-query_id \t neighbour_document_id \t pseudo-query_text \t neighbour_document_text`,
as seen in the script `prepare_text_pairs.py`.
```
1. Convert to features for BERT input:
python convert_pairs_to_features_csv.py --input_text_pairs_file ${path_to_query-doc_pairs_file} \
                                        --output_features_file ${path_to_output_features_file} \
                                        --max_seq_length 512
2. Run BERT inference:
bash run_bert_pairs_score.sh
```



### 2. Online Retrieval
For each input query, we first use BM25 to get its seed documents (top-30/50 in our experiments).
The pseudo-queries and neighbour documents of seed documents are used to online retrieval.
Then, we use BERT ranker to calculate the similarity scores from query to pseudo-queries (of seed documents).
This online query-(pseudo)query matching is less expensive than query-document cross-attentions.

Firstly, we need to prepare a text file that contains query-(pseudo)query pairs, like 
`query_id \t pseudo-query_id \t query_text \t pseudo-query_text`, as seen in the script `prepare_text_pairs.py`.
After that, we convert text to features and feed them into BERT ranker for computing the similarity scores.
```
1. Convert to features for BERT input:
python convert_pairs_to_features_csv.py --input_text_pairs_file ${path_to_query_pairs_file} \
                                        --output_features_file ${path_to_output_features_file} \
                                        --max_seq_length 128
2. Run BERT inference: 
bash run_bert_pairs_score.sh
```

Meanwhile, we also utilize exact lexicon match signals to supplement the final relevance scores. 
Herein, we use [Pyserini](https://github.com/castorini/pyserini/) toolkit to compute BM25 scores of 
the expanded neighbour document to the input query, using the corpus index built by Anserini. 
This step can be done during query-(pseudo)query matching, without introducing any online cost.
```
% Obtain bm25 scores for query-neighbour pairs from expanded corpus index
python read_bm25_from_index.py --query_file ${path_to_query_file} \
                               --init_bm25_file ${path_to_init_bm25_file} \
                               --seed_doc_num 50 \
                               --neighbour_doc_file ${path_to_neighbours_file} \
                               --output_q_pgqs_bm25_file ${path_to_q_pgqs_bm25_file} \
                               --index_dir ${path_to_anserini_index}
```

Lastly, we re-score all neighbour documents using query-(pseudo)query similarity and (pseudo)query-document 
relevance, and re-rank these neighbour documents for the input query and output the final ranking list.
You can find the script `rank_neighbour_docs.sh` for an example.
```
% Rank query-neighbour pairs and return final rank file
python rank_neighbour_docs.py --init_bm25_file ${path_to_init_bm25_file} \
                              --seed_doc_num 50 \
                              --neighbour_doc_file ${path_to_neighbours_file} \
                              --qgq_score_file ${path_to_qgq_score_file} \
                              --gqp_score_file ${path_to_gqp_score_file} \
                              --q_pgqs_bm25_file ${path_to_q_pgqs_bm25_file} \
                              --alpha_weight 0.9 \
                              --hits_num 1000 \
                              --output_rank_file ${path_to_output_rank_file}
```