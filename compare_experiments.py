import json

with open("experiment_v1.jsonl", "r", encoding="utf-8") as file:
    k3 = json.load(file)

with open("experiment_v2.jsonl", "r", encoding="utf-8") as file:
    k1 = json.load(file)

with open("experiment_v3.jsonl", "r", encoding="utf-8") as file:
    k2 = json.load(file)


print("\n========== ALL COMPARISON ==========\n")

print(f"{'Metric':<22} {'k=3':<10} {'k=2':<10} {'k1=':<10}")
print("-" * 65)


for metric in k3["metrics"]:
    score_k3 = k3["metrics"][metric]
    score_k2 = k2["metrics"][metric]
    score_k1 = k1["metrics"][metric]

    print(
        f"{metric.replace('_', ' ').title():<22} "
        f"{score_k3:<15.2f} "
        f"{score_k2:<15.2f} "
        f"{score_k1:+.2f}"
    )