import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from data.preprocess import (
    get_dataloader,
    prepare_test_datasets,
    prepare_train_datasets,
)
from dotenv import load_dotenv

load_dotenv()


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

TEST_DATA_PATH = ""

model_path = ""
model_name = ""
batch_size = 1
temperature = 1.0
K = 2

model = AutoModelForCausalLM.from_pretrained(
    model_path, use_auth_token=HUGGING_FACE_TOKEN
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGING_FACE_TOKEN)

test_dataset = prepare_test_datasets(TEST_DATA_PATH, tokenizer)

test_dataloader = get_dataloader(test_dataset, batch_size)

total_accuracy = 0

for batch in test_dataloader:

    x, y, I, answer = batch["x"], batch["y"], batch["I"], batch["answer"]

    input_text = f"{x}\n{y}\n{I}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    scores = []
    for _ in range(K):

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            yes_token_id = tokenizer.encode("Yes")[0]
            yes_prob = probs[0, yes_token_id].item()
            scores.append(yes_prob)

    avg_score = sum(scores) / K
    final_label = ["Yes"] if avg_score > 0.5 else ["No"]

    total_accuracy += (final_label == answer).item()

print(total_accuracy)
