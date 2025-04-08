import os
import json
from rouge import Rouge
from tqdm import tqdm
import argparse
from baseline import BM25Retriever, DenseRetriever
from llm import LLMGenerator

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

def load_gnn_data(gnn_file, val_data):
    with open(gnn_file, "r") as f:
        gnn_data = json.load(f)
    data = []
    for item in val_data:
        query = item["query"]
        if query in gnn_data:
            data.append({
                "query": query,
                "answer": item["answer"],
                "retrieved_passages": gnn_data[query]
            })
        else:
            print(f"Warning: Query not found in GNN data: {query}")
    return data

def get_retriever_and_index(all_passages, model_type, dense_model_name):
    bm25_retriever = BM25Retriever(all_passages)
    bm25_retriever.index(True)
    
    dense_retriever = DenseRetriever(all_passages, dense_model_name)
    dense_retriever.index(True)

    if model_type == "bm25":
        return bm25_retriever
    else:
        return dense_retriever

def evaluate_retrieval(retriever, dataset, k=10):
    mrr_scores_10, precision_scores_10, f1_scores_10 = [], [], []
    mrr_scores_5, precision_scores_5, f1_scores_5 = [], [], []
    global_offset = 0

    for item in tqdm(dataset, desc="Evaluating retrieval"):
        question = item["query"]
        gold_labels = item["gold_labels"]
        gold_labels_idx = [global_offset + j for j, label in enumerate(gold_labels) if label == 1]

        _, _, top_indices = retriever.retrieve(question, k=k)
        
        for cutoff, mrr_scores, precision_scores, f1_scores in [
            (10, mrr_scores_10, precision_scores_10, f1_scores_10),
            (5, mrr_scores_5, precision_scores_5, f1_scores_5)
        ]:
            # MRR and F1
            relevant_ranks = []
            num_relevant_retrived = 0

            for i, idx in enumerate(top_indices[:cutoff]):
                if idx in gold_labels_idx:
                    relevant_ranks.append(i + 1)
                    num_relevant_retrived += 1
            
            if relevant_ranks:
                mrr_scores.append(1.0 / min(relevant_ranks))
                
                recall = num_relevant_retrived / len(gold_labels_idx) if gold_labels_idx else 0
                precision = num_relevant_retrived / cutoff
                precision_scores.append(precision)
                
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                f1_scores.append(f1)
            else:
                mrr_scores.append(0)
                precision_scores.append(0)
                f1_scores.append(0)

        global_offset += len(gold_labels) 

    def avg(lst): return sum(lst) / len(lst) if lst else 0

    retrieval_metrics = {
        "MRR@10": avg(mrr_scores_10),
        "Precision@10": avg(precision_scores_10),
        "F1@10": avg(f1_scores_10),
        "MRR@5": avg(mrr_scores_5),
        "Precision@5": avg(precision_scores_5),
        "F1@5": avg(f1_scores_5),
    }
    
    print("Retrieval metrics:", retrieval_metrics)
    return retrieval_metrics

def evaluate_generation(generator, retriever, dataset, model_type, k=10):
    rouge = Rouge()
    f1_scores_10, rouge_scores_10, em_scores_10 = [], [], []
    f1_scores_5, rouge_scores_5, em_scores_5 = [], [], []
    global_offset = 0
    detailed_logs = []

    os.makedirs("retrieval_logs", exist_ok=True)
    for idx, item in enumerate(tqdm(dataset, desc="Evaluating generation")):
        question = item["query"]
        gold_answer = item["answer"]
        gold_labels = item["gold_labels"]
        gold_labels_idx = [global_offset + j for j, label in enumerate(gold_labels) if label == 1]
        global_offset += len(gold_labels) 

        top_passages, _, _ = retriever.retrieve(question, k=k)

        generated_answer_5 = generator.generate(question, top_passages[:5])
        detailed_logs.append({
            "query": question,
            "gold_answer": gold_answer,
            "generated_answer_5": generated_answer_5,
            "retrieved_passages_5": top_passages[:5],
            "gold_passages": [retriever.passages[idx] for idx in gold_labels_idx]
        })

        # F1 score (word overlap) for top-5
        pred_words_5 = set(generated_answer_5.lower().split())
        gold_words_5 = set(gold_answer.lower().split())
        common_5 = len(pred_words_5.intersection(gold_words_5))
        precision_5 = common_5 / len(pred_words_5) if pred_words_5 else 0
        recall_5 = common_5 / len(gold_words_5) if gold_words_5 else 0
        f1_5 = 2 * precision_5 * recall_5 / (precision_5 + recall_5) if (precision_5 + recall_5) > 0 else 0
        f1_scores_5.append(f1_5)

        # Exact match for top-5
        em_5 = int(generated_answer_5.strip().lower() == gold_answer.strip().lower())
        em_scores_5.append(em_5)

        # ROUGE for top-5
        try:
            rouge_score_5 = rouge.get_scores(generated_answer_5, gold_answer)[0]
            rouge_scores_5.append(rouge_score_5)
        except:
            rouge_scores_5.append({
                "rouge-1": {"f": 0, "p": 0, "r": 0},
                "rouge-2": {"f": 0, "p": 0, "r": 0},
                "rouge-l": {"f": 0, "p": 0, "r": 0}
            })

        # Evaluate for top-10 passages
        generated_answer_10 = generator.generate(question, top_passages[:10])
        detailed_logs.append({
            "query": question,
            "gold_answer": gold_answer,
            "generated_answer_10": generated_answer_10,
            "retrieved_passages_10": top_passages[:10],
            "gold_passages": [retriever.passages[idx] for idx in gold_labels_idx]
        })

        # F1 score (word overlap) for top-10
        pred_words_10 = set(generated_answer_10.lower().split())
        gold_words_10 = set(gold_answer.lower().split())
        common_10 = len(pred_words_10.intersection(gold_words_10))
        precision_10 = common_10 / len(pred_words_10) if pred_words_10 else 0
        recall_10 = common_10 / len(gold_words_10) if gold_words_10 else 0
        f1_10 = 2 * precision_10 * recall_10 / (precision_10 + recall_10) if (precision_10 + recall_10) > 0 else 0
        f1_scores_10.append(f1_10)

        # Exact match for top-10
        em_10 = int(generated_answer_10.strip().lower() == gold_answer.strip().lower())
        em_scores_10.append(em_10)

        # ROUGE for top-10
        try:
            rouge_score_10 = rouge.get_scores(generated_answer_10, gold_answer)[0]
            rouge_scores_10.append(rouge_score_10)
        except:
            rouge_scores_10.append({
                "rouge-1": {"f": 0, "p": 0, "r": 0},
                "rouge-2": {"f": 0, "p": 0, "r": 0},
                "rouge-l": {"f": 0, "p": 0, "r": 0}
            })

    with open(f"retrieval_logs/all_query_logs_{model_type}.json", "w") as f:
        json.dump(detailed_logs, f, indent=2)

    avg_f1_5 = sum(f1_scores_5) / len(f1_scores_5) if f1_scores_5 else 0
    avg_rouge_5 = {
        "rouge-1": {"f": 0, "p": 0, "r": 0},
        "rouge-2": {"f": 0, "p": 0, "r": 0},
        "rouge-l": {"f": 0, "p": 0, "r": 0}
    }
    avg_em_5 = sum(em_scores_5) / len(em_scores_5)

    if rouge_scores_5:
        for score in rouge_scores_5:
            for k, v in score.items():
                for metric, value in v.items():
                    avg_rouge_5[k][metric] += value / len(rouge_scores_5)

    avg_f1_10 = sum(f1_scores_10) / len(f1_scores_10) if f1_scores_10 else 0
    avg_rouge_10 = {
        "rouge-1": {"f": 0, "p": 0, "r": 0},
        "rouge-2": {"f": 0, "p": 0, "r": 0},
        "rouge-l": {"f": 0, "p": 0, "r": 0}
    }
    avg_em_10 = sum(em_scores_10) / len(em_scores_10)

    if rouge_scores_10:
        for score in rouge_scores_10:
            for k, v in score.items():
                for metric, value in v.items():
                    avg_rouge_10[k][metric] += value / len(rouge_scores_10)

    generation_metrics = {
        "F1@5": avg_f1_5,
        "Exact_Match@5": avg_em_5,
        "ROUGE-1-F@5": avg_rouge_5["rouge-1"]["f"],
        "ROUGE-2-F@5": avg_rouge_5["rouge-2"]["f"],
        "ROUGE-L-F@5": avg_rouge_5["rouge-l"]["f"],
        "F1@10": avg_f1_10,
        "Exact_Match@10": avg_em_10,
        "ROUGE-1-F@10": avg_rouge_10["rouge-1"]["f"],
        "ROUGE-2-F@10": avg_rouge_10["rouge-2"]["f"],
        "ROUGE-L-F@10": avg_rouge_10["rouge-l"]["f"]
    }

    print("Generation metrics:", generation_metrics)
    return generation_metrics

def evaluate_gnn_generation(generator, dataset):
    rouge = Rouge()
    f1_scores_5, rouge_scores_5, em_scores_5 = [], [], []
    f1_scores_10, rouge_scores_10, em_scores_10 = [], [], []
    detailed_logs = []

    os.makedirs("retrieval_logs", exist_ok=True)
    for item in tqdm(dataset, desc="Evaluating GNN generation"):
        question = item["query"]
        gold_answer = item["answer"]
        retrieved_passages = item["retrieved_passages"]

        # Evaluate for top-5 passages
        generated_answer_5 = generator.generate(question, retrieved_passages[:5])
        detailed_logs.append({
            "query": question,
            "gold_answer": gold_answer,
            "generated_answer_5": generated_answer_5,
            "retrieved_passages_5": retrieved_passages[:5]
        })

        # F1 score for top-5
        pred_words_5 = set(generated_answer_5.lower().split())
        gold_words_5 = set(gold_answer.lower().split())
        common_5 = len(pred_words_5.intersection(gold_words_5))
        precision_5 = common_5 / len(pred_words_5) if pred_words_5 else 0
        recall_5 = common_5 / len(gold_words_5) if gold_words_5 else 0
        f1_5 = 2 * precision_5 * recall_5 / (precision_5 + recall_5) if (precision_5 + recall_5) > 0 else 0
        f1_scores_5.append(f1_5)

        # Exact match for top-5
        em_5 = int(generated_answer_5.strip().lower() == gold_answer.strip().lower())
        em_scores_5.append(em_5)

        # ROUGE for top-5
        try:
            rouge_score_5 = rouge.get_scores(generated_answer_5, gold_answer)[0]
            rouge_scores_5.append(rouge_score_5)
        except:
            rouge_scores_5.append({
                "rouge-1": {"f": 0, "p": 0, "r": 0},
                "rouge-2": {"f": 0, "p": 0, "r": 0},
                "rouge-l": {"f": 0, "p": 0, "r": 0}
            })

        # Evaluate for top-10 passages
        generated_answer_10 = generator.generate(question, retrieved_passages[:10])
        detailed_logs.append({
            "query": question,
            "gold_answer": gold_answer,
            "generated_answer_10": generated_answer_10,
            "retrieved_passages_10": retrieved_passages[:10]
        })

        # F1 score for top-10
        pred_words_10 = set(generated_answer_10.lower().split())
        gold_words_10 = set(gold_answer.lower().split())
        common_10 = len(pred_words_10.intersection(gold_words_10))
        precision_10 = common_10 / len(pred_words_10) if pred_words_10 else 0
        recall_10 = common_10 / len(gold_words_10) if gold_words_10 else 0
        f1_10 = 2 * precision_10 * recall_10 / (precision_10 + recall_10) if (precision_10 + recall_10) > 0 else 0
        f1_scores_10.append(f1_10)

        # Exact match for top-10
        em_10 = int(generated_answer_10.strip().lower() == gold_answer.strip().lower())
        em_scores_10.append(em_10)

        # ROUGE for top-10
        try:
            rouge_score_10 = rouge.get_scores(generated_answer_10, gold_answer)[0]
            rouge_scores_10.append(rouge_score_10)
        except:
            rouge_scores_10.append({
                "rouge-1": {"f": 0, "p": 0, "r": 0},
                "rouge-2": {"f": 0, "p": 0, "r": 0},
                "rouge-l": {"f": 0, "p": 0, "r": 0}
            })

    with open("retrieval_logs/all_query_logs_gnn.json", "w") as f:
        json.dump(detailed_logs, f, indent=2)

    # Compute averages for top-5
    avg_f1_5 = sum(f1_scores_5) / len(f1_scores_5) if f1_scores_5 else 0
    avg_rouge_5 = {
        "rouge-1": {"f": 0, "p": 0, "r": 0},
        "rouge-2": {"f": 0, "p": 0, "r": 0},
        "rouge-l": {"f": 0, "p": 0, "r": 0}
    }
    avg_em_5 = sum(em_scores_5) / len(em_scores_5)

    if rouge_scores_5:
        for score in rouge_scores_5:
            for k, v in score.items():
                for metric, value in v.items():
                    avg_rouge_5[k][metric] += value / len(rouge_scores_5)

    # Compute averages for top-10
    avg_f1_10 = sum(f1_scores_10) / len(f1_scores_10) if f1_scores_10 else 0
    avg_rouge_10 = {
        "rouge-1": {"f": 0, "p": 0, "r": 0},
        "rouge-2": {"f": 0, "p": 0, "r": 0},
        "rouge-l": {"f": 0, "p": 0, "r": 0}
    }
    avg_em_10 = sum(em_scores_10) / len(em_scores_10)

    if rouge_scores_10:
        for score in rouge_scores_10:
            for k, v in score.items():
                for metric, value in v.items():
                    avg_rouge_10[k][metric] += value / len(rouge_scores_10)

    # Generate final metrics
    generation_metrics = {
        "F1@5": avg_f1_5,
        "Exact_Match@5": avg_em_5,
        "ROUGE-1-F@5": avg_rouge_5["rouge-1"]["f"],
        "ROUGE-2-F@5": avg_rouge_5["rouge-2"]["f"],
        "ROUGE-L-F@5": avg_rouge_5["rouge-l"]["f"],
        "F1@10": avg_f1_10,
        "Exact_Match@10": avg_em_10,
        "ROUGE-1-F@10": avg_rouge_10["rouge-1"]["f"],
        "ROUGE-2-F@10": avg_rouge_10["rouge-2"]["f"],
        "ROUGE-L-F@10": avg_rouge_10["rouge-l"]["f"]
    }

    print("Generation metrics:", generation_metrics)
    return generation_metrics

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

    if args.model_type != "gnn":
        retrieval_metrics = evaluate_retrieval(retriever, val_data, k=args.top_k)
        generation_metrics = evaluate_generation(generator, retriever, val_data, args.model_type, k=args.top_k)
    else:
        gnn_data = load_gnn_data(args.gnn_file, val_data)
        generation_metrics = evaluate_gnn_generation(generator, gnn_data)
        retrieval_metrics = {}

    save_results(args, retrieval_metrics, generation_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN-based Retrieval for QA")
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results and models")
    parser.add_argument("--gnn_file", type=str, default="./data/gnn_retrieved.json",
                    help="Path to GNN retrieved passages file (used only if model_type=gnn)")
    parser.add_argument("--model_type", type=str, default="bm25",
                        choices=["bm25", "dense", "gnn"],
                        help="Retriever model type")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-mpnet-base-v2",
                        help="Sentence embedding model for dense retriever")
    parser.add_argument("--llm_model", type=str, default="google/flan-t5-base",
                        help="Language model for answer generation")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of passages to retrieve")
    args = parser.parse_args()
    main(args)
