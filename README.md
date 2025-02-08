# Introduction 
An Efficient Cardinality Estimation method (**BPF**) with Query-induced Probability
 Inference.
# Environment
Details in environment.yaml file.
# Datasets
Using the [End-to-End-Benchmark](https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark) for test. Following the instruction to load the LJOB and STATS dataset.

The CEB and JOB query workload are in the sqls folder.
# Code Explaination
1. BPF related codes
   - model definition (model folder)
   - learning function (learning folder)
   - load data function for batch training(data folder)
   - loss function (func folder)
2. SQL query workload
   - LJOB query workload (sqls/job_sub_query.sql)
   - STATS query workload (sqls/stats_sub_query.sql)
3. Pre-processing of Datasets
   - add header of IMDB Dataset (prepare_imdb_header.py)
   - preprocess function of dataset like data bucketing (preprocess_csv.py)
   - parse SQL query to Select-Project-Join (query_parse.py)
   - train BPF on every table of datasets (train_bpf.py)
   - the main functions to train BPF (train_func.py)
   - the main fucntions for inference (inference_func.py)
   - single table Cardinality Estimation test (single_table_ce.py)
   - multi table Cardinality Estimation test (multi_table_ce.py)
4. Auxiliary classes
   - AttributePair (attr_pair.py) for join attribute of table.
   - BasicInfo (basic_info.py) for some useful information of table.

# Run
```
python train_bpf.py

python single_table_ce.py

python multi_table_ce.py
```