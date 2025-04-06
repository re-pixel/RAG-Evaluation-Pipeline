import pandas as pd
import json
import chromadb
import numpy as np
from utils import *

class Evaluation():

    def __init__(self, corpus_name, corpus_path, questions_path, chroma_db_path):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.corpus = f.read()
        self.questions_df = pd.read_csv(questions_path)
        self.questions_df = self.questions_df[self.questions_df['corpus_id'] == corpus_name]
        self.questions_df['references'] = self.questions_df['references'].apply(json.loads)
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)

    def _get_chunks_with_intervals(self, chunker):
        chunks = chunker.split_text(self.corpus)
        chunk_intervals = []
        for chunk in chunks:
            _, start, end = rigorous_document_search(self.corpus, chunk)
            chunk_intervals.append({"start_index": start, "end_index": end})
        
        return chunks, chunk_intervals
    
    def _total_evaluation(self, chunk_intervals):
        iou_scores = []
        useful_chunks_count = []

        for _, row in self.questions_df.iterrows():
            references = row['references']

            iou_score = 0
            useful_chunk_count = 0
            numerator_sets = []
            denominator_chunks_sets = []
            unused_highlights = [(x['start_index'], x['end_index']) for x in references]

            for interval in chunk_intervals:
                useful = False
                
                for reference in references:
                    
                    ref_start, ref_end = int(reference['start_index']), int(reference['end_index'])
                    intersec = intersection((interval['start_index'], interval['end_index']), (ref_start, ref_end))
                    
                    if intersec is not None:
                        useful = True
                        unused_highlights = difference(unused_highlights, intersec)
                        numerator_sets = union([intersec] + numerator_sets)                        
                        denominator_chunks_sets = union([(interval['start_index'], interval['end_index'])] + denominator_chunks_sets)
            
                if useful:
                    useful_chunk_count += 1
                
            useful_chunks_count.append(useful_chunk_count)
            denominator_sets = union(denominator_chunks_sets + unused_highlights)
            if numerator_sets:
                iou_score = sum_of_ranges(numerator_sets) / sum_of_ranges(denominator_sets)
            
            iou_scores.append(round(iou_score, 3))

        return iou_scores, useful_chunks_count
    
    def _evaluate_retrieval(self, chunk_intervals_by_question):
        iou_scores = []
        recall_scores = []
        precision_scores = []

        for (_, row), intervals in zip(self.questions_df.iterrows(), chunk_intervals_by_question):
            references = row['references']

            numerator_sets = []
            denominator_chunks_sets = []
            unused_highlights = [(x['start_index'], x['end_index']) for x in references]

            for interval in intervals:
                chunk_start, chunk_end = interval['start_index'], interval['end_index']

                for ref in references:
                    ref_start, ref_end = int(ref['start_index']), int(ref['end_index'])
                    
                    intersec = intersection((chunk_start, chunk_end), (ref_start, ref_end))
                    
                    if intersec is not None:
                        unused_highlights = difference(unused_highlights, intersec)
                        numerator_sets = union([intersec] + numerator_sets)
                        denominator_chunks_sets = union([(chunk_start, chunk_end)] + denominator_chunks_sets)
            

            if numerator_sets:
                numerator_value = sum_of_ranges(numerator_sets)
            else:
                numerator_value = 0

            recall_denominator = sum_of_ranges([(x['start_index'], x['end_index']) for x in references])
            precision_denominator = sum_of_ranges([(x['start_index'], x['end_index']) for x in intervals])
            iou_denominator = precision_denominator + sum_of_ranges(unused_highlights)

            recall_score = numerator_value / recall_denominator
            recall_scores.append(recall_score)

            precision_score = numerator_value / precision_denominator
            precision_scores.append(precision_score)

            iou_score = numerator_value / iou_denominator
            iou_scores.append(iou_score)

        return iou_scores, recall_scores, precision_scores

            
    def _chunks_to_collection(self, chunker, embedding_function, collection_name):
        collection = self.chroma_client.get_or_create_collection(collection_name, embedding_function=embedding_function, metadata={"hnsw:search_ef":50})
            
        chunks, intervals = self._get_chunks_with_intervals(chunker)

        BATCH_SIZE = 500
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[i:i+BATCH_SIZE]
            batch_intervals = intervals[i:i+BATCH_SIZE]
            batch_ids = [str(i) for i in range(i, i+len(batch_chunks))]
            collection.add(
                documents=batch_chunks,
                metadatas=batch_intervals,
                ids=batch_ids
            )

        return collection

    def _questions_to_collection(self, embedding_function):
        collection = self.chroma_client.get_or_create_collection("questions", embedding_function=embedding_function, metadata={"hnsw:search_ef":50})
        collection.add(
            documents=self.questions_df['question'].tolist(),
            ids=[str(i) for i in self.questions_df.index]
        )
        return collection

    def run(self, chunker, embedding_function, number_of_chunks_to_retrieve):
        chunks_collection_name = embedding_function.name + '_' + str(chunker._chunk_size) + '_' + str(chunker._chunk_overlap) + '_' + str(number_of_chunks_to_retrieve)
        chunks_collection = self._chunks_to_collection(chunker, embedding_function, chunks_collection_name)

        questions_collection = self._questions_to_collection(embedding_function)
        question_db = questions_collection.get(include=['embeddings'])

        question_db['ids'] = [int(id) for id in question_db['ids']]
        _, sorted_embeddings = zip(*sorted(zip(question_db['ids'], question_db['embeddings'])))

        self.questions_df = self.questions_df.sort_index()

        #brute_force_iou, _ = self._total_evaluation(chunks_collection.get()['metadatas'])
        retrievals = chunks_collection.query(query_embeddings=list(sorted_embeddings), n_results=number_of_chunks_to_retrieve)
        iou_scores, recall_scores, precision_scores = self._evaluate_retrieval(retrievals['metadatas'])

        # brute__iou_mean = np.mean(brute_force_iou)
        # brute_iou_std = np.std(brute_force_iou)

        recall_mean = round(np.mean(recall_scores)*100, 1)
        recall_std = round(np.std(recall_scores)*100, 1)

        iou_mean = round(np.mean(iou_scores)*100, 1)
        iou_std = round(np.std(iou_scores)*100, 1)

        precision_mean = round(np.mean(precision_scores)*100, 1)
        precision_std = round(np.std(precision_scores)*100, 1)

        return recall_mean, recall_std, iou_mean, iou_std, precision_mean, precision_std