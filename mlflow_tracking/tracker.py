import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

EXPERIMENT_NAME     = "omnimind-rag"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


class OmniMindTracker:
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
            mlflow.create_experiment(EXPERIMENT_NAME)

        mlflow.set_experiment(EXPERIMENT_NAME)
        print(f"[mlflow] Tracking → {MLFLOW_TRACKING_URI}")

    def log_rag_config(self, chunk_size, chunk_overlap, k, model_name):
        with mlflow.start_run(run_name="rag-config"):
            mlflow.log_params({
                "chunk_size":      chunk_size,
                "chunk_overlap":   chunk_overlap,
                "retrieval_k":     k,
                "llm_model":       model_name,
                "embedding_model": "all-MiniLM-L6-v2",
            })

    def log_chat(self, question, answer, source_count, response_time):
        with mlflow.start_run(run_name="chat", nested=True):
            mlflow.log_metrics({
                "response_time_ms": response_time,
                "answer_length":    len(answer),
                "source_count":     source_count,
                "question_length":  len(question),
            })

    def log_vision(self, task, processing_time, result_count=0, confidence=0.0):
        with mlflow.start_run(run_name=f"vision-{task}", nested=True):
            mlflow.log_param("task_type", task)
            mlflow.log_metrics({
                "processing_time_ms": processing_time,
                "result_count":       result_count,
                "confidence":         confidence,
            })


def run_demo():
    import random
    tracker = OmniMindTracker()

    tracker.log_rag_config(
        chunk_size=800, chunk_overlap=100,
        k=4, model_name="llama-3.1-8b-instant"
    )

    for i in range(5):
        tracker.log_chat(
            question      = f"Test question {i}",
            answer        = "Test answer " * random.randint(5, 20),
            source_count  = random.randint(1, 4),
            response_time = random.uniform(200, 1500),
        )

    print("Done! Run:  mlflow ui --port 5001")
    print("Then open: http://localhost:5001")


if __name__ == "__main__":
    run_demo()