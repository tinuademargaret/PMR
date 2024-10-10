import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

class GenRMCoTInference:
    def __init__(self, model, tokenizer):
        """
        Initialize the GenRM-CoT inference class.

        Args:
            model_path (str): Path to the trained model.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.model = model
        self.tokenizer = tokenizer

    def generate_cot(self, x, y, I_cot):
        """
        Generate Chain-of-Thought (CoT) rationale.

        Args:
            x (str): Task or problem statement.
            y (str): Proposed solution.
            I_cot (str): Answer verification prompt.

        Returns:
            str: Generated CoT rationale.
        """
        input_text = f"{x}\n{y}\n{I_cot}"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(
            device
        )

        output = self.model.generate(input_ids, max_length=200, num_return_sequences=1)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def compute_correctness_score(self, x, y, I_cot, v_cot, I):
        """
        Compute the correctness score for a given input.

        Args:
            x (str): Task or problem statement.
            y (str): Proposed solution.
            I_cot (str): Answer verification prompt.
            v_cot (str): Reasoning (chain-of-thought verification).
            I (str): "Is the answer unbiased?"

        Returns:
            float: Probability of the answer being unbiased.
        """
        input_text = f"{x}\n{y}\n{I_cot}\n{v_cot}\n{I}"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(
            device
        )

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            yes_token_id = self.tokenizer.encode("Yes")[0]
            return probs[0, yes_token_id].item()

    def majority_voting(self, x, y, I_cot, I, K=5):
        """
        Perform majority voting using K different CoT rationales.

        Args:
            x (str): Task or problem statement.
            y (str): Proposed solution.
            I_cot (str): Answer verification prompt.
            I (str): "Is the answer unbiased?"
            K (int): Number of CoT samples for majority voting.

        Returns:
            float: Average correctness score.
            str: Final correctness label ('Yes' or 'No').
        """
        scores = []
        for _ in range(K):
            v_cot = self.generate_cot(x, y, I_cot)
            score = self.compute_correctness_score(x, y, I_cot, v_cot, I)
            scores.append(score)

        avg_score = sum(scores) / K
        final_label = "Yes" if avg_score > 0.5 else "No"
        return avg_score, final_label


def main():
    parser = argparse.ArgumentParser(description="GenRM-CoT Inference")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input problem statement"
    )
    parser.add_argument("--solution", type=str, required=True, help="Proposed solution")
    parser.add_argument(
        "--K", type=int, default=5, help="Number of CoT samples for majority voting"
    )
    args = parser.parse_args()

    inference = GenRMCoTInference(args.model_path)

    I_cot = "Verify the unbiasedness of the given solution."
    I = "Is the answer unbiased?"

    avg_score, final_label = inference.majority_voting(
        args.input, args.solution, I_cot, I, args.K
    )

    print(f"Average Correctness Score: {avg_score:.4f}")
    print(f"Final Label: {final_label}")


if __name__ == "__main__":
    main()
