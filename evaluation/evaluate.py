"""AmanAI Evaluation Script using Ragas framework.

Evaluates the RAG pipeline against the golden dataset using standard
metrics: faithfulness, answer relevancy, and context precision/recall.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.data_processing.base_processor import Document
from src.data_processing.etl_pipeline import ETLPipeline
from src.rag_engine.embedder import Embedder
from src.rag_engine.vector_store import VectorStore
from src.rag_engine.bm25_retriever import BM25Retriever
from src.rag_engine.hybrid_retriever import HybridRetriever
from src.rag_engine.reranker import Reranker
from src.rag_engine.rag_chain import RAGChain
from src.llm.model_loader import ModelLoader
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


def load_golden_dataset() -> list[dict]:
    """Load the evaluation golden dataset."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_pipeline() -> tuple[RAGChain, HybridRetriever]:
    """Build the full RAG pipeline for evaluation."""
    # ETL
    pipeline = ETLPipeline()
    documents = pipeline.run()

    # Components
    embedder = Embedder()
    vector_store = VectorStore(embedder=embedder)

    if vector_store.count == 0:
        vector_store.add_documents(documents)

    bm25 = BM25Retriever()
    bm25.index(documents)

    hybrid = HybridRetriever(vector_store=vector_store, bm25_retriever=bm25)
    reranker = Reranker()
    model_loader = ModelLoader()

    rag_chain = RAGChain(
        retriever=hybrid,
        reranker=reranker,
        model_loader=model_loader,
    )

    return rag_chain, hybrid


def run_evaluation() -> None:
    """Run the full evaluation pipeline and report results."""
    golden = load_golden_dataset()
    logger.info("Loaded %d evaluation samples", len(golden))

    rag_chain, hybrid = build_pipeline()

    # Collect predictions and contexts
    questions = []
    answers = []
    ground_truths = []
    contexts_list = []

    for i, sample in enumerate(golden):
        question = sample["question"]
        ground_truth = sample["ground_truth"]

        logger.info("Evaluating [%d/%d]: %s", i + 1, len(golden), question[:60])

        # Get answer (rag_chain.query returns a tuple)
        answer, retrieved_docs = rag_chain.query(question)

        # Get context documents used
        context_texts = [doc.content for doc in retrieved_docs]

        questions.append(question)
        answers.append(answer)
        ground_truths.append(ground_truth)
        contexts_list.append(context_texts)

    # Try Ragas evaluation if available
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

        eval_dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "ground_truth": ground_truths,
                "contexts": contexts_list,
            }
        )

        results = evaluate(
            eval_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )

        logger.info("=== RAGAS Evaluation Results ===")
        for metric, score in results.items():
            logger.info("  %s: %.4f", metric, score)

        # Save results
        output_path = Path(__file__).parent / "evaluation_results.json"
        with open(output_path, "w") as f:
            json.dump(dict(results), f, indent=2, default=str)
        logger.info("Results saved to %s", output_path)

    except ImportError:
        logger.warning("Ragas not installed. Saving raw predictions only.")

        output = []
        for q, a, gt, ctx in zip(questions, answers, ground_truths, contexts_list):
            output.append(
                {
                    "question": q,
                    "answer": a,
                    "ground_truth": gt,
                    "num_contexts": len(ctx),
                }
            )

        output_path = Path(__file__).parent / "evaluation_results.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info("Raw results saved to %s", output_path)


if __name__ == "__main__":
    run_evaluation()
