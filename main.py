from chunker import FixedTokenChunker
from evaluation import Evaluation
from embedding import EmbeddingFunction
import pandas as pd

chunker_parameters = [(100, 0),
                      (200, 0),
                      (400, 0),
                      (800, 0),
                      (400, 200),
                      (800, 400)]

n_retrievals = [1, 5, 10]

results_df = pd.read_csv("results/results.csv", index_col=0)

with open("data/wikitexts.md", "r", encoding='utf-8') as f:
    text = f.read()

evaluation = Evaluation('wikitexts', 'data/wikitexts.md', 'data/questions_df.csv', './chromadb')
embedding_function = EmbeddingFunction()

for (chunk_size, overlap_size) in chunker_parameters:
    chunker = FixedTokenChunker(chunk_size=chunk_size, chunk_overlap=overlap_size)
    for n in n_retrievals:
        result = results_df[(results_df['chunk_size'] == chunk_size)
                            & (results_df['overlap'] == overlap_size)
                            & (results_df['retrieved_chunks'] == n)]
        if result.empty:
            recall_mean, recall_std, iou_mean, iou_std, precision_mean, precision_std = evaluation.run(chunker, embedding_function, n)
            new_row = pd.DataFrame([{"chunk_size": chunk_size, "overlap": overlap_size, "retrieved_chunks": n,
                                "recall": str(recall_mean) + '±' + str(recall_std),
                                "IOU": str(iou_mean) + '±' + str(iou_std), 
                                "precision": str(precision_mean) + '±' + str(precision_std)}])
            results_df = pd.concat([results_df, new_row], ignore_index=True)

results_df = results_df.sort_values(by=['retrieved_chunks', 'overlap', 'chunk_size']).reset_index(drop=True)
results_df.to_csv("results/results.csv")