from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
import json
from groq_judge import GroqJudge

test_case = LLMTestCase(
    input="Where do penguins live?",
    actual_output="Penguins live in Maharashtra and sometimes live in the Southern Hemisphere.",
    expected_output="Penguins live in the Southern Hemisphere, including Antarctica, South America, southern Africa, and the Galapagos Islands."
)

metric = GEval(
    name="Correctness",
    criteria=(
        "Determine whether the actual output is factually correct based on the expected output"
    ),
    evaluation_params=[
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT
    ],
    model=GroqJudge("openai/gpt-oss-120b")
)

metric.measure(test_case=test_case)

print("Score: ", metric.score)
print("Reason: ", metric.reason)
