## Bidirectional Tracing

The project is the experimental code for article "Verifiable Bidirectional Traceability Queries for Multi-Source Blockchain Databases". This paper proposes a novel bidirectional traceability index for blockchain systems, designed to support both forward and backward traceability queries efficiently. For forward traceability, the index is designed  based on hash aggregation to compress the verification object (VO). For backward traceability, we partition the whole data flow graph into time-based subgraphs and introduce the Sorted Merkle Patricia Trie (SMPT) to construct the cross-subgraph index, therefore improving backward tracing efficiency. Additionally, we propose an active data replication mechanism to further compress the VO size and minimize verification costs, enabling more efficient traceability queries. The security analysis proves that the proposed scheme achieves verifiable tracing. Experimental results show the effectiveness of the work.


## Environments, necessary libraries or tools
- Python 3.9
- cytoolz 0.9.0.1
- eth-hash 0.2.0
- eth-typing 2.0.0
- eth-utils 1.4.1
- pycryptodome 3.19.1
- rlp 1.1.0
- toolz 0.9.0

## Usage
1. Download the code to your directory.
2. Download datasets: cit-HepTh.txt, cit-HepTh-dates.txt from Hep-Th (https://snap.stanford.edu/data/cit-HepTh.html).  In the root directory, unzip the file.
3. Run filter.py to generate small datasets or run the whole dataset. You may need to choose the file path and name in the code. 
4. Excute the code. main.py is to execute bidirectional tracing. multi.py is to execute multi-keyword tracing. time.py is to execute time-range tracing.

## License
By contributing, you agree that your contributions will be licensed under its MIT License.


