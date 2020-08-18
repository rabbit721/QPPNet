# QPPNet in Pytorch

This code contains a sample implementation for Plan-Structured Deep Neural Network Models for Query Performance Prediction(https://arxiv.org/pdf/1902.00132.pdf), written for the purpose of comparison with MB2: Decomposed Behavior Modeling for Self-Driving Database 
Management Systems on the TPC-H benchmark, the TPC-C benchmark, and our self-generated dataset smallbank.

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

### Examples for Training a model

- On TPC-H benchmark dataset generated with SF=1

  ``` 
  python3 main.py --dataset TPCH -s 0 -t 250000 --batch_size 128 -epoch_freq 1000 --lr 5e-3 --step_size 1000 --SGD --scheduler --data_dir ./tpchlarge2/900-exp_res_by_temp/ --num_q 22 --num_sample_per_q 900
  ```

- On TPC-C benchmark dataset

  ``` 
  python3 main.py --dataset TPCC -s 0 -t 250000 --batch_size 512 -epoch_freq 1000 --lr 2e-3 --step_size 1000 --SGD --scheduler --data_dir ./dataset/tpcc_dataset/tpcc_pipeline.csv
  ```

### Examples for Testing a trained model

- Testing a model trained on TPC-H benchmark (SF=1) dataset for 4000 epochs on TPC-H benchmark (SF=10) dataset

  ``` 
  python3 main.py --test_time --dataset TPCH -s 4000 --data_dir ./dataset/tpch_dataset/tpch10G/900-exp_res_by_temp/
  ```

- Testing a model trained on the TPC-C benchmark for 10000 epochs on the smallbank dataset

  ```
  python3 main.py --test_time --dataset TPCC -s 10000 --data_dir ./dataset/tpcc_dataset/sb_pipeline.csv
  ```

  

