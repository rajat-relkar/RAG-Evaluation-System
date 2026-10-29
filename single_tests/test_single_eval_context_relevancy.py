from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from groq_judge import GroqJudge

test_case = LLMTestCase(
    input="Where do penguins live?",
    actual_output="Penguins live in a wide range of environments in the Southern Hemisphere, including the icy coasts of Antarctica, the cool shores of South America and southern Africa, and even the Galapagos Islands near the equator.",
    retrieval_context=["Penguins are stupid."]
)

metric = ContextualRelevancyMetric(threshold=0.7, model=GroqJudge("openai/gpt-oss-120b"))

metric.measure(test_case)

print("Score: ", metric.score)
print("Reason: ", metric.reason)
print("Passed: ", metric.is_successful())