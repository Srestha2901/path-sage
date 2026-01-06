# 1. Introduction

Path-SAGE is a temporal network analysis framework designed to identify influential nodes in aggregated temporal networks. This repository provides the algorithms created either to run the state-of-the-art models and the ones that implement Path-SAGE. It also provides the datasets used in the evaluation experiments.

To cite this work: Sadhu, S., Namtirtha, A., C. Malta, M., Dutta, A. (2025). PathSAGE: Identifying Influential Spreaders in Temporal Networks with GraphSAGE. *IEEE Transactions on Emerging Topics in Computational Intelligence* (accepted for publication)

# 2. Installation & Requirements

Before running the code, ensure that all required Python Libraries are installed. They are:
> matplotlib ~= 3.9.0  
> numpy ~= 1.26.4  
> networkx ~= 3.2.1  
> torch ~= 2.3.1  
> scipy ~= 1.13.1  
> pandas ~= 2.2.3  
> scikit-learn ~= 1.5.0

# 3. Clone the Repository
git clone https://github.com/your-username/path-sage.git
cd path-sage

# 4. How to Run the Code
## 4.1. Load the Dataset

All datasets used in Path-SAGE evaluation are provided inside the "Datasets" folder.

## 4.2. Preprocessing

Run the following scripts located in the "util" folder:

> ReadGraph.py
> 
> DistanceMatrix.py

These scripts parse the temporal graph and generate the necessary distance matrices.

## 4.3. State-of-the-Art Models

The baseline methods used for comparison are available inside the folder "algorithms.state-of-the-art"

## 4.4. Proposed Method Implementation

The Path-SAGE framework comprises two algorithms, as described in the paper (Algorithms 1 and 2).
Both algorithms are implemented together in a single file located in the folder "Proposed_method"

## 4.5. Experimental Results

All the experiment results are available in the "Results" folder. This folder includes:

> Experiment1.xls – Contains results for Experiment 1
> 
> Experiment 2 & 3.xls – Contains results for both Experiment 2 and Experiment 3 (each dataset has its own sheet)
> 
> Experiment4.xls – Results for Experiment 4
> 
> Experiment5.xls – Results for Experiment 5
