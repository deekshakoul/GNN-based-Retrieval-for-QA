import os
import json
from rouge import Rouge
from tqdm import tqdm
import argparse
from baseline import BM25Retriever, DenseRetriever
from llm import LLMGenerator

def evaluate_retrieval(retriever, dataset, k=10):
    mrr_scores = []
    precision_scores = []
    f1_scores = []
    global_offset = 0

    for item in tqdm(dataset, desc="Evaluating retrieval"):
        question = item["query"]
        gold_labels = item["gold_labels"]
        gold_labels_idx = [global_offset + j for j, label in enumerate(gold_labels) if label == 1]

        _, _, top_indices = retriever.retrieve(question, k=k)
        
        # MRR and F1
        relevant_ranks = []
        num_relevant_retrived = 0

        for i, idx in enumerate(top_indices):
            if idx in gold_labels_idx:
                relevant_ranks.append(i + 1)
                num_relevant_retrived += 1
        
        if relevant_ranks:
            mrr_scores.append(1.0 / min(relevant_ranks))
            
            recall = num_relevant_retrived / len(gold_labels_idx) if gold_labels_idx else 0
            precision = num_relevant_retrived / k
            precision_scores.append(precision)
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        else:
            mrr_scores.append(0)
            precision_scores.append(0)
            f1_scores.append(0)

        global_offset += len(gold_labels) 

    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    retrieval_metrics = {
        "MRR": avg_mrr,
        "F1": avg_f1
    }
    
    print("Retrieval metrics:", retrieval_metrics)
    return retrieval_metrics

def evaluate_generation(generator, retriever, dataset, k=10):
    rouge = Rouge()
    f1_scores = []
    rouge_scores = []
    em_scores = []

    for item in tqdm(dataset, desc="Evaluating generation"):
        question = item["query"]
        gold_answer = item["answer"]
        
        top_passages, _, _ = retriever.retrieve(question, k=k)
        generated_answer = generator.generate(question, top_passages)

        # F1 score (word overlap)
        pred_words = set(generated_answer.lower().split())
        gold_words = set(gold_answer.lower().split())
        common = len(pred_words.intersection(gold_words))
        precision = common / len(pred_words) if pred_words else 0
        recall = common / len(gold_words) if gold_words else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

        # Exact match
        em = int(generated_answer.strip().lower() == gold_answer.strip().lower())
        em_scores.append(em)
        
        # ROUGE
        try:
            rouge_score = rouge.get_scores(generated_answer, gold_answer)[0]
            rouge_scores.append(rouge_score)
        except:
            rouge_scores.append({
                "rouge-1": {"f": 0, "p": 0, "r": 0},
                "rouge-2": {"f": 0, "p": 0, "r": 0},
                "rouge-l": {"f": 0, "p": 0, "r": 0}
            })
    
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    avg_rouge = {
        "rouge-1": {"f": 0, "p": 0, "r": 0},
        "rouge-2": {"f": 0, "p": 0, "r": 0},
        "rouge-l": {"f": 0, "p": 0, "r": 0}
    }
    avg_em = sum(em_scores) / len(em_scores)
    
    if rouge_scores:
        for score in rouge_scores:
            for k, v in score.items():
                for metric, value in v.items():
                    avg_rouge[k][metric] += value / len(rouge_scores)
    
    generation_metrics = {
        "F1": avg_f1,
        "Exact_Match": avg_em,
        "ROUGE-1-F": avg_rouge["rouge-1"]["f"],
        "ROUGE-2-F": avg_rouge["rouge-2"]["f"],
        "ROUGE-L-F": avg_rouge["rouge-l"]["f"]
    }

    print("Generation metrics:", generation_metrics)
    return generation_metrics

def load_data(data_path, split="dev"):
    file_path = os.path.join(data_path, f"{split}.json")
    with open(file_path, "r") as f:
        val_dataset = json.load(f)

    all_passages = []
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
        val_data.append({"query": query, "gold_labels": gold_labels, "answer": answer})
    return all_passages, val_data

def get_retriever_and_index(all_passages, model_type, dense_model_name):
    bm25_retriever = BM25Retriever(all_passages)
    bm25_retriever.index(True)
    
    dense_retriever = DenseRetriever(all_passages, dense_model_name)
    dense_retriever.index(True)

    if model_type == "bm25":
        return bm25_retriever
    else:
        return dense_retriever

def save_results(args, retrieval_metrics, generation_metrics):
    results = {
        "model_type": args.model_type,
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": generation_metrics
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"results_{args.model_type}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

def main(args):
    all_passages, val_data = load_data(args.data_path)
    
    retriever = get_retriever_and_index(all_passages, args.model_type, args.embedding_model)
    generator = LLMGenerator(model_name=args.llm_model)

    retrieval_metrics = evaluate_retrieval(retriever, val_data, k=args.top_k)
    generation_metrics = evaluate_generation(generator, retriever, val_data, k=args.top_k)
    save_results(args, retrieval_metrics, generation_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN-based Retrieval for QA")
    parser.add_argument("--data_path", type=str, default="/raid/infolab/deekshak/ir3/data/data_ids/",
                        help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results and models")
    parser.add_argument("--model_type", type=str, default="bm25",
                        choices=["bm25", "dense"],
                        help="Retriever model type")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-mpnet-base-v2",
                        help="Sentence embedding model for dense retriever")
    parser.add_argument("--llm_model", type=str, default="t5-base",
                        help="Language model for answer generation")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of passages to retrieve")
    args = parser.parse_args()
    main(args)
