# RAG Evaluation Pipeline

A comprehensive evaluation pipeline for Retrieval-Augmented Generation (RAG) systems, designed to assess and optimize chunking strategies for document retrieval.

## Overview

This project provides a systematic framework for evaluating different text chunking and retrieval strategies in RAG systems. It uses ChromaDB for vector storage and retrieval, and evaluates performance across multiple metrics including recall, precision, and Intersection over Union (IoU). More detailed overview is given in the report/report.ipynb file.

## Features

- **Multiple Chunking Strategies**: Evaluate different chunk sizes and overlap configurations
- **Comprehensive Metrics**: Assess retrieval quality using recall, precision, and IoU scores
- **Vector Database Integration**: ChromaDB-based storage for efficient similarity search over embeddings
- **Configurable Retrieval**: Test different numbers of retrieved chunks
- **Batch Processing**: Efficient processing of large document collections

## Repository Structure

```
.
├── chunker.py          # Text chunking implementations (FixedTokenChunker)
├── embedding.py        # Embedding function wrapper
├── evaluation.py       # Main evaluation class and metrics
├── main.py            # Pipeline entry point and experiment runner
├── utils.py           # Helper functions for text search and range operations
├── data/              # Dataset files
│   ├── wikitexts.md          # Sample corpus for evaluation
│   └── questions_df.csv      # Questions and reference answers
├── results/           # Evaluation results and visualizations
│   ├── results.csv           # Aggregated evaluation metrics
│   └── table.py             # Script to generate results visualization
└── report/            # Analysis and documentation
    └── report.ipynb          # Detailed analysis notebook
```

## Key Evaluation Metrics

- **Recall**: Measures how much of the relevant information is retrieved
- **Precision**: Measures the relevance of retrieved information
- **IoU (Intersection over Union)**: Measures overlap between retrieved and reference text ranges

## Results Folder

The `results/` directory serves as a reference for evaluation outcomes:

### `results.csv`
Contains comprehensive evaluation metrics across different configurations:
- **chunk_size**: Token count per chunk (100, 200, 400, 800)
- **overlap**: Number of overlapping tokens between chunks (0, 200, 400)
- **retrieved_chunks**: Number of chunks retrieved per query (1, 5, 10)
- **recall**: Mean ± std deviation of recall scores
- **IOU**: Mean ± std deviation of Intersection over Union scores
- **precision**: Mean ± std deviation of precision scores

### `results_table.png` (Generated)
Visual representation of the results table. This file is auto-generated from `results.csv` using `table.py`.

## Getting Started

### Prerequisites

```bash
pip install pandas chromadb tiktoken fuzzywuzzy matplotlib
```

### Running the Evaluation

1. Place your corpus in `data/` directory
2. Create a questions dataset with references in CSV format
3. Configure chunking parameters in `main.py`
4. Run the evaluation:

```bash
python main.py
```

The evaluation will:
- Create chunks using various configurations
- Store embeddings in ChromaDB (local `./chromadb/` directory)
- Retrieve relevant chunks for each question
- Calculate metrics and update `results/results.csv`

### Generating Results Visualization

```bash
cd results
python table.py
```

This generates `results_table.png` from the current `results.csv`.

## Configuration

Main evaluation parameters in `main.py`:

```python
chunker_parameters = [(100, 0), (200, 0), (400, 0), (800, 0), (400, 200), (800, 400)]
n_retrievals = [1, 5, 10]
```

Modify these to test different chunking strategies and retrieval configurations.

## Data Format

### Questions CSV
Expected format:
- `corpus_id`: Identifier for the corpus
- `question`: The question text
- `references`: JSON array of reference text ranges

### Corpus
Markdown or text format containing the source documents for evaluation.
