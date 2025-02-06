# BPF
A Cardinality Estimation Method Based on BPF.
# Environment
Details in environment.yaml file.
# Datasets
Using the [End-to-End-Benchmark](https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark) for test. Following the instruction to load the LJOB and STATS dataset.

The CEB and JOB query workload are in the sqls folder.
# Run
python train_bpf.py
python single_table_ce.py
python multi_table_ce.py