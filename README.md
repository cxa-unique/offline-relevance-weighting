# Contextualized Offline Relevance Weighting
This repository contains some resources of our paper:
- [Contextualized Offline Relevance Weighting for Efficient and Effective Neural Retrieval](https://dl.acm.org/doi/pdf/10.1145/3404835.3463073). In *SIGIR 2021*.

## Framework
1. Offline System

![image](https://github.com/cxa-unique/offline-relevance-weighting/blob/main/offline_system.bmp)

2. Online System

![image](https://github.com/cxa-unique/offline-relevance-weighting/blob/main/online_system.bmp)

## Resources
1. Selected neighbour documents<sup>*</sup>: 
   * TREC-19-DL-Passage: [Download](https://drive.google.com/file/d/1HeDNMc_g6-yPey9t8ZIe83cqj6cy30RZ/view?usp=sharing)
   * TREC-20-DL-Passage: [Download](https://drive.google.com/file/d/1UDVDMTNjdfBdW5-Yc2Il3fb-4XBrTIQD/view?usp=sharing)
   * TREC-19-DL-Document: [Download](https://drive.google.com/file/d/1UcPctPsa80CuK3oFl_sILG-qLD7bgoQO/view?usp=sharing)
   * TREC-20-DL-Document: [Download](https://drive.google.com/file/d/1Z7wZYOnGFLtTE8nyIZPYXcfOWGEywufP/view?usp=sharing)
   * Format: `seed_doc_id \t neighbour_doc_id \t rank \t recall_frequency \t best_bm25_score \n`
2. The final relevance scores of all recalled neighbour documents<sup>*</sup>:
   * TREC-19-DL-Passage: [Download](https://drive.google.com/file/d/1yYtk5vOoCDYGRQPYOjWvn2X_VvL7R6W5/view?usp=sharing)
   * TREC-20-DL-Passage: [Download](https://drive.google.com/file/d/1S9CPnwrhy7ddemnTnhqEtS2VUdva8riS/view?usp=sharing)
   * TREC-19-DL-Document: [Download](https://drive.google.com/file/d/1VEumyY7VSsj5ebWvPcOIcS_7ug4yJv-I/view?usp=sharing)
   * TREC-20-DL-Document: [Download](https://drive.google.com/file/d/1zLyi3b1BNzDW3GnyiX-niQBt7eSdgxJJ/view?usp=sharing)
   * Format: `query_id \t neighbour_doc_id \t rank \t rel_score \n`
   
   <sup>*</sup> Note that the neighbour documents here belong to the 
   top-100 (**s=100**) seed documents by BM25, and the `neighbour document` for 
   document ranking task refers to the passages segmented from documents
   using the script `convert_msmarco_passages_doc_to_anserini.py` in 
   [docTTTTTquery](https://github.com/castorini/docTTTTTquery) repo.
3. Retrieval run files (**top-1,000**):
   * TREC-19-DL-Passage: [BM25](https://drive.google.com/file/d/1_AyvVVbcGesSwg98ULcvi1QByRBATO9i/view?usp=sharing), 
   [Ours(s=30)](https://drive.google.com/file/d/1IlrbgscRpxefcEpNSegImDpfZlTSsnWm/view?usp=sharing), 
   [Ours(s=50)](https://drive.google.com/file/d/17M-CJd5G6fcqQ1DodnYKzNhlzuD4tNSA/view?usp=sharing).
   * TREC-20-DL-Passage: [BM25](https://drive.google.com/file/d/1zPqTqWEjD9WbAo40WOFznfJNlpjq57Ud/view?usp=sharing), 
   [Ours(s=30)](https://drive.google.com/file/d/1BK9NG7_bPw_6HD3rxLrNxxnFKRuFvJ04/view?usp=sharing), 
   [Ours(s=50)](https://drive.google.com/file/d/1pEeGQXeEXLvNnxtyN9iVtwaEyARSUhvq/view?usp=sharing).
   * TREC-19-DL-Document: [BM25](https://drive.google.com/file/d/1LyVmDVDUg9Zd6cUKYx35uZ_4-GlMQehz/view?usp=sharing), 
   [Ours(s=30)](https://drive.google.com/file/d/1KCBYBX5X6R190eJIMT2G3VqL9zgpFrli/view?usp=sharing), 
   [Ours(s=50)](https://drive.google.com/file/d/1wu-uBhaiJ0iJD9Bl9Hu8CRnd427hyRP9/view?usp=sharing).
   * TREC-20-DL-Document: [BM25](https://drive.google.com/file/d/1FW9mlM6Sfro7cwdiRKp-kd7SmpFxW4Gi/view?usp=sharing), 
   [Ours(s=30)](https://drive.google.com/file/d/1k0ET2ehcAlhqhl1MZfOkal7rJvIKlAuN/view?usp=sharing), 
   [Ours(s=50)](https://drive.google.com/file/d/1F7URsZLxek9dK7r8DPydx3LoQfz9FbWq/view?usp=sharing).
   * Format: `query_id Q0 doc_id rank score run_name`
   
For more details of our method and experiments, please refer to our paper.

## Citation
If you find our paper/resources useful, please cite: 
```
@inproceedings{Chen2021_sigir,
 author = {Xuanang Chen and
           Ben He and
           Kai Hui and
           Yiran Wang and
           Le Sun and
           Yingfei Sun},
 title = {Contextualized Offline Relevance Weighting for Efficient and Effective Neural Retrieval},
 booktitle = {Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
 pages = {1617-1621},
 year = {2021},
}
```