Path-SAGE

Path-SAGE is a temporal network analysis framework designed to identify influential nodes in aggregated temporal networks. 

Citations to this work: XXXXXXX

# Installation & Requirements

Before running the code, ensure that all required Python Libraries are installed. They are:


> matplotlib ~= 3.9.0  
> numpy ~= 1.26.4  
> networkx ~= 3.2.1  
> torch ~= 2.3.1  
> scipy ~= 1.13.1  
> pandas ~= 2.2.3  
> scikit-learn ~= 1.5.0

# Clone the Repository
git clone https://github.com/your-username/path-sage.git
cd path-sage

# How to Run the Code
## 1. Load the Dataset

All datasets required for running the framework are provided inside the Dataset/ folder.

## 2. Preprocessing

Run the following scripts located in the util/ directory:

> ReadGraph.py
> DistanceMatrix.py

These scripts parse the temporal graph and generate the necessary distance matrices.

## 3. Competitive State-of-the-Art Models

The baseline methods used for comparison are available inside:

> algorithms.state-of-the-art/

## 4. Proposed Method Implementation

The proposed Path-SAGE framework comprises two algorithms, as described in the paper (Algorithms 1 and 2).
Both algorithms are implemented together in a single file located in the folder "Proposed_method"

## 5. Experimental Results

All experiment outputs are stored inside the Results/ directory. The result files include:

> Experiment1.xls – Contains results for Experiment 1
> Experiment 2 & 3.xls – Contains results for both Experiment 2 and Experiment 3 (each dataset has its own sheet)
> Experiment4.xls – Results for Experiment 4
> Experiment5.xls – Results for Experiment 5
