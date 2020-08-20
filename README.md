# QPPNet in Pytorch

This code contains a sample implementation for [Plan-Structured Deep Neural Network Models for Query Performance Prediction](https://arxiv.org/pdf/1902.00132.pdf), and the code for training/testing on

- TPC-H queries generated using https://github.com/gregrahn/tpch-kit.git and benchmarked with Postgres

- TPC-H queries generated using https://github.com/gregrahn/tpch-kit.git and benchmarked with Terrier

- TPC-C queries and smallbank dataset generated using [OLTP](https://github.com/oltpbenchmark/oltpbench) and benchmarked with [Terrier](https://github.com/cmu-db/terrier).

## Prerequisites

- Linux or macOS
- Python 3

## Getting Started

### Installation

- Cloning the repo:

  ```
  git clone https://github.com/rabbit721/QPPNet.git
  cd QPPNet
  ```

- Install the required python packages:
  - For pip: `pip install -r requirements.txt`
  - For conda: `conda env create -f environment.yml`

### Getting the Dataset from the Web Directory

- TPC-H benchmarked with PostgresSQL

  ```
  # Under directory datasets/postgres_tpch_dataset
  wget http://www.andrew.cmu.edu/user/jiejiao/data/qpp/postgres/tpch/psqltpch0p1g.zip && unzip psqltpch0p1g.zip
  wget http://www.andrew.cmu.edu/user/jiejiao/data/qpp/postgres/tpch/psqltpch1g.zip && unzip psqltpch1g.zip
  wget http://www.andrew.cmu.edu/user/jiejiao/data/qpp/postgres/tpch/psqltpch10g.zip && unzip psqltpch10g.zip
  ```

- TPC-H benchmarked with Terrier:
  
  Data files already located under directory [`datasets/terrier_tpch_dataset`](https://github.com/rabbit721/QPPNet/tree/master/dataset/terrier_tpch_dataset) as `execution_0p1G.csv`, `execution_1G.csv`, and `execution_10G.csv`

- TPC-C and smallbank benchmarked with Terrier:
  
  Data files already located under directory [`datasets/oltp_dataset`](https://github.com/rabbit721/QPPNet/tree/master/dataset/terrier_tpch_dataset) as `tpcc_pipeline.csv` and `sb_pipeline.csv`

### Examples for Training a model

- On TPC-H dataset generated using https://github.com/gregrahn/tpch-kit.git with SF=1 and benchmarked with PostgresSQL

  ```
  python3 main.py --dataset PSQLTPCH -s 0 -t 250000 --batch_size 128 -epoch_freq 1000 --lr 2e-3 --step_size 1000 --SGD --scheduler --data_dir ./dataset/postgres_tpch_dataset/tpch1g/900-exp_res_by_temp/ --num_q 22 --num_sample_per_q 900
  ```

- On TPC-H dataset generated using https://github.com/gregrahn/tpch-kit.git with SF=1 and benchmarked with Terrier

  ```
  python3 main.py --dataset TerrierTPCH -s 0 -t 250000 --batch_size 512 -epoch_freq 1000 --lr 1e-3 --step_size 1000 --SGD --scheduler --data_dir ./dataset/terrier_dataset/execution_1G.csv
  ```

- On TPC-C dataset generated using OLTP with SF=1 and benchmarked with Terrier

  ```
  python3 main.py --dataset OLTP -s 0 -t 250000 --batch_size 512 -epoch_freq 1000 --lr 5e-3 --step_size 1000 --SGD --scheduler --data_dir ./dataset/oltp_dataset/tpcc_pipeline.csv
  ```

### Examples for Testing a trained model

- Testing a model trained on TPC-H SF=1 dataset benchmarked with PostgresSQL for 4000 epochs on TPC-H SF=10 dataset benchmarked with PostgresSQL.
  Please make sure that models are saved in `./saved_model`

  ```
  python3 main.py --test_time --dataset PSQLTPCH -s 4000 --data_dir ./dataset/postgres_tpch_dataset/tpch10G/900-exp_res_by_temp/
  ```

- Testing a model trained on the TPC-C benchmark for 10000 epochs on the smallbank dataset.
  Please make sure that models are saved in `./saved_model`

  ```
  python3 main.py --test_time --dataset OLTP -s 10000 --data_dir ./dataset/oltp_dataset/sb_pipeline.csv
  ```
