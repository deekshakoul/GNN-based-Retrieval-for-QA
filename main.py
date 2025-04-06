import os
import json
from rouge import Rouge
from tqdm import tqdm
import argparse
from baseline import BM25Retriever, DenseRetriever
from llm import LLMGenerator

def evaluate_retrieval(retriever, dataset, k=10):
    mrr_scores = []
    recall_at_k = []
    global_offset = 0
    for i, item in tqdm(dataset, desc="Evaluating retrieval"):
        question = item["query"]
        gold_labels = item["gold_labels"]
        gold_labels_idx = [global_offset + j for j,label in enumerate(gold_labels) if label==1]

        _, _, top_indices = retriever.retrieve(question, k=k)
        
        # Calculate MRR and Recall@k
        relevant_ranks = []
        for i, idx in enumerate(top_indices):
            if idx in gold_labels_idx:
                relevant_ranks.append(i + 1)
        
        if relevant_ranks:
            # MRR: reciprocal of the rank of the first relevant passage
            mrr_scores.append(1.0 / min(relevant_ranks))
            
            # Recall@k: whether at least one relevant passage is retrieved
            recall_at_k.append(1 if len(relevant_ranks) > 0 else 0)

        global_offset += len(gold_labels) 

    # Calculate average metrics
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
    avg_recall = sum(recall_at_k) / len(recall_at_k) if recall_at_k else 0
    
    return {
        "MRR": avg_mrr,
        f"Recall@{k}": avg_recall
    }


def evaluate_generation(generator, retriever, dataset, k=10):
    rouge = Rouge()
    f1_scores = []
    rouge_scores = []
    
    for item in tqdm(dataset, desc="Evaluating generation"):
        question = item["query"]
        gold_answer = item["answer"]
        
        top_passages, _, _ = retriever.retrieve(question, k=k)
        generated_answer = generator.generate(question, top_passages)

        # Calculate F1 score (word overlap)
        pred_words = set(generated_answer.lower().split())
        gold_words = set(gold_answer.lower().split())
        common = len(pred_words.intersection(gold_words))
        precision = common / len(pred_words) if pred_words else 0
        recall = common / len(gold_words) if gold_words else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
        
        # Calculate ROUGE scores
        try:
            rouge_score = rouge.get_scores(generated_answer, gold_answer)[0]
            rouge_scores.append(rouge_score)
        except:
            # Handle edge cases (empty strings, etc.)
            pass
    
    # Calculate average metrics
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    # Average ROUGE scores
    avg_rouge = {
        "rouge-1": {"f": 0, "p": 0, "r": 0},
        "rouge-2": {"f": 0, "p": 0, "r": 0},
        "rouge-l": {"f": 0, "p": 0, "r": 0}
    }
    
    if rouge_scores:
        for score in rouge_scores:
            for k, v in score.items():
                for metric, value in v.items():
                    avg_rouge[k][metric] += value / len(rouge_scores)
    
    return {
        "F1": avg_f1,
        "ROUGE-1-F": avg_rouge["rouge-1"]["f"],
        "ROUGE-2-F": avg_rouge["rouge-2"]["f"],
        "ROUGE-L-F": avg_rouge["rouge-l"]["f"]
    }

def main(args):
    file_path = os.path.join("data", "dev.json")
    with open(file_path, "r") as f:
        val_dataset = json.load(f)
    all_passages = []
    # all_gold = []
    val_data = []
    for i, item in enumerate(val_dataset):
        query = item["question"]
        answer = item["answer"]
        passages = item["context"]
        gold_labels = [0] * len(passages)
        for j, passage in enumerate(passages):
            title = passage[0]
            sentences = passage[1]
            passage_str = title + "\n" + "\n".join(sentences)

            supporting_facts_titles = [title for title, _ in item["supporting_facts"]]
            if title in supporting_facts_titles:
                gold_labels[j] = 1
            
            all_passages.append(passage_str)
        # all_gold.append(gold_labels)
        val_data.append({"query": query, "gold_labels": gold_labels, "answer": answer})

    bm25_retriever = BM25Retriever()
    bm25_retriever.index(all_passages)
    
    dense_retriever = DenseRetriever()
    dense_retriever.index(all_passages)
    
    if args.model_type == "bm25":
        active_retriever = bm25_retriever
    else:
        active_retriever = dense_retriever
    
    generator = LLMGenerator(model_name=args.llm_model)
    
    print("Evaluating retrieval performance...")
    retrieval_metrics = evaluate_retrieval(active_retriever, val_data, k=args.top_k)
    print("Retrieval metrics:", retrieval_metrics)
    
    print("Evaluating answer generation...")
    generation_metrics = evaluate_generation(generator, active_retriever, val_data, k=args.top_k)
    print("Generation metrics:", generation_metrics)
    
    results = {
        "model_type": args.model_type,
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": generation_metrics
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"results_{args.model_type}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN-based Retrieval for QA")
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results and models")
    parser.add_argument("--model_type", type=str, default="bm25",
                        choices=["bm25", "dense"],
                        help="Retriever model type")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["retrieval", "generation", "analysis", "all"],
                        help="Evaluation mode")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence embedding model for dense retriever")
    parser.add_argument("--llm_model", type=str, default="t5-base",
                        help="Language model for answer generation")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of passages to retrieve")
    args = parser.parse_args()
    
    main(args)