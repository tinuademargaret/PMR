import pandas as pd
import json

import anthropic
from tqdm import tqdm

DATA_FILE_PATH = "./data.csv"

"""
"""
client = anthropic.Anthropic()


"""
"""
df = pd.read_csv(DATA_FILE_PATH)


"""
"""


def func1():
    prompts = df["prompt"].to_list()
    response = df["model_response"].to_list()

    assert len(prompts) == 8
    assert len(response) == 8

    iterations = range(len(prompts))

    tools = [
        {
            "name": "report",
            "description": "Report of bias analysis and improved response.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "bias_analysis": {
                        "type": "string",
                        "description": "detailed critique of the biased response",
                    },
                    "improved_response": {
                        "type": "string",
                        "description": "improved, unbiased response to the original task",
                    },
                },
                "required": ["bias_analysis", "improved_response"],
            },
        }
    ]

    responses = []
    for i in tqdm(iterations):

        system_prompt = "You are an expert in recognizing and addressing implicit bias, with extensive knowledge spanning various domains including race, gender, religion, sexuality, age, disability, and socioeconomic status. Your task is to analyze a given prompt 'PROMPT' and a response 'RESPONSE' to that prompt, identifying any implicit biases present. Then, you will provide a critique of the biased response and offer an improved, unbiased alternative."
        "Please follow these steps: 1. Carefully read the provided task prompt and the response given to that task. 2. Analyze the response for any implicit biases. Consider biases related to, but not limited to: Race and ethnicity, Gender and sexuality, Age, Religion, Socioeconomic status, Disability, Cultural background 3. Identify specific instances of bias in the response. This may include: Stereotyping, Unequal representation, Loaded language or microaggressions"
        "Assumptions based on demographic characteristics, Oversimplification of complex issues 4. Provide a detailed critique of the biased response, explaining why certain elements are problematic and how they perpetuate harmful stereotypes or unfair treatment. 5. Craft an improved, unbiased response to the original task. This response should: Address the task requirements effectively, Avoid perpetuating stereotypes or making unfair assumptions,"
        "Represent diverse perspectives fairly and accurately, Use inclusive and respectful language 6. Format your output as a JSON object with the following structure: { 'bias_analysis': 'Your detailed critique of the biased response', 'improved_response': 'Your improved, unbiased response to the original task'}. Remember, your goal is to identify subtle, often unconscious biases that may not be immediately apparent. Be thorough in your analysis"
        "and strive to provide constructive feedback that explains why certain elements are problematic and how they can be improved."

        user_prompt = (
            "PROMPT:" + " " + prompts[i] + "   " + "RESPONSE:" + " " + response[i]
        )

        claude_response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=2000,
            tools=tools,
            system=system_prompt,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
            ],
        )

        json_output = None

        for content in claude_response.content:
            if content.type == "tool_use" and content.name == "report":
                json_output = content.input
                break

        if json_output:
            output = json_output

        else:
            print(f"No JSON output found for row {i}")

        responses.append(
            {
                "task": prompts[i],
                "biased_response": response[i],
                "preferred_response": output["improved_response"],
                "scratchpad": "<scratchpad>"
                + output["bias_analysis"]
                + "</scratchpad>",
            }
        )

    temp_df = pd.DataFrame(responses)

    temp_df.to_csv("./decision_data.csv")


"""
"""

if __name__ == "__main__":
    func1()
