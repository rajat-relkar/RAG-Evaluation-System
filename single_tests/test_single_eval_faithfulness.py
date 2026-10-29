from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from groq_judge import GroqJudge

test_case = LLMTestCase(
    input="Where do penguins live?",
    actual_output="Penguins live in a wide range of environments in the Southern Hemisphere, including the icy coasts of Antarctica, the cool shores of South America and southern Africa, and even the Galapagos Islands near the equator.",
    retrieval_context=["Penguins are flightless seabirds that are specially adapted for life in the Southern Hemisphere. Although people often associate them only with Antarctica, penguins live in a wide range of environments, including the icy coasts of Antarctica, the cool shores of South America and southern Africa, and even the Galapagos Islands near the equator."]
)

metric = FaithfulnessMetric(threshold=0.7, model=GroqJudge("openai/gpt-oss-120b")) # or use smaller model - openai/gpt-oss-20b

metric.measure(test_case)

print("Score: ", metric.score)
print("Reason: ", metric.reason)
print("Passed: ", metric.is_successful())