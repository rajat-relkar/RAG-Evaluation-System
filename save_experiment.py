import json
import sys
from pathlib import Path
from statistics import mean

if len(sys.argv) != 3:
    sys.exit("Usage: python save_experiment.py <version> <retriever_k>   e.g. python save_experiment.py 2 1")

version = sys.argv[1]
retriever_k = int(sys.argv[2])

file_name = f"evaluated_results{version}.jsonl"

# a fresh run writes next to where it was started; the checked-in copies live in versions/vN
candidates = [Path(file_name), Path(__file__).parent / "versions" / f"v{version}" / file_name]
input_path = next((path for path in candidates if path.exists()), None)

if input_path is None:
    sys.exit(f"Could not find {file_name} in the current folder or in versions/v{version}/")

with open(input_path, "r", encoding="utf-8") as file:
    results = [json.loads(line) for line in file]

experiment = {
    "experiment": f"experiment_v{version}_k{retriever_k}",
    "retriever_k": retriever_k,
    "number_of_test_cases": len(results),

    "metrics": {
        "answer_relevancy": mean(r["answer_relevancy"] for r in results),
        "faithfulness": mean(r["faithfulness"] for r in results),
        "context_relevancy": mean(r["context_relevancy"] for r in results),
        "correctness": mean(r["correctness"] for r in results),
    }
}

output_path = f"experiment_v{version}.jsonl"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(experiment, file, indent=4)

print(f"Experiment saved to {output_path} ...")
