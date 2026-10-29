from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from groq_judge import GroqJudge

test_case = LLMTestCase(
    input="What do penguins eat?",
    actual_output=("Penguins primarily feed on cars.")
)
# Penguins primarily feed on marine animals such as krill and tiny fish

metric = AnswerRelevancyMetric(threshold=0.7, model=GroqJudge("openai/gpt-oss-20b"))

metric.measure(test_case)

print("Score: ", metric.score)
print("Reason: ", metric.reason)
print("Passed: ", metric.is_successful())