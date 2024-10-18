# %%
import argparse
import sys
import os
from dataclasses import dataclass
import torch
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)

import wandb

from data.preprocess import (
    get_dataloader,
    prepare_test_datasets,
    prepare_train_datasets,
)

from inference import GenRMCoTInference

from dotenv import load_dotenv

load_dotenv()


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

TRAIN_DATA_PATH = "./data/decision_data.csv"
VALIDATION_DATA_PATH = "./data/decision_data.csv"


# %%
@dataclass
class GenRMCoTTrainerConfig:
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    lambda_param: float = 1.0
    learning_rate: float = 5e-5
    num_epochs: int = 10
    batch_size: int = 1


class GenRMCoTTrainer:
    def __init__(self, config: GenRMCoTTrainerConfig):
        """
        Initialize the GenRM-CoT trainer.

        Args:
            model_name (str): Name of the pre-trained model to use.
            lambda_param (float): Lambda parameter for balancing losses.
            learning_rate (float): Learning rate for optimization.
            num_epochs (int): Number of training epochs.
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name, use_auth_token=HUGGING_FACE_TOKEN
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, use_auth_token=HUGGING_FACE_TOKEN
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        wandb.init(
            project="GenRM-CoT",
            config={
                "learning_rate": config.learning_rate,
                "epochs": config.num_epochs,
            },
        )

        # Get datasets
        cot_train_dataset, correct_train_dataset = prepare_train_datasets(
            TRAIN_DATA_PATH, self.tokenizer
        )

        validation_dataset = prepare_test_datasets(VALIDATION_DATA_PATH, self.tokenizer)

        # Get dataloaders
        self.cot_dataloader = get_dataloader(cot_train_dataset, config.batch_size)
        self.correct_dataloader = get_dataloader(
            correct_train_dataset, config.batch_size
        )
        self.validation_dataloader = get_dataloader(
            validation_dataset, config.batch_size
        )

        self.lambda_param = config.lambda_param
        self.learning_rate = config.learning_rate
        self.num_epochs = config.num_epochs
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.total_steps = len(self.cot_dataloader) * self.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.total_steps
        )
        self.validation_inference = GenRMCoTInference(config.model_name, self.tokenizer)

    def validation_step(self, batch):
        """
        Perform validation on a batch.

        Args:
            batch: Validation batch.

        Returns:
            torch.Tensor: Validation loss.
        """
        output_label = self.validation_inference.majority_voting(
            batch["x"], batch["y"], batch["I_cot"], batch["I"]
        )
        target = batch["answer"]
        return output_label == target

    def train_step(self, batch):
        """
        Perform a training step on a batch.

        Args:
            batch: Training batch.

        Returns:
            torch.Tensor: Training loss.
        """

        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = self.model(**inputs)
        return outputs.loss

    def train(self):
        """
        Train the GenRM-CoT model.

        Args:
            cot_dataloader (DataLoader): DataLoader for the CoT dataset.
            correct_dataloader (DataLoader): DataLoader for the correct solutions dataset.
        """

        self.best_loss = float("inf")
        self.best_epoch = 0
        self.patience = 5

        for epoch in range(self.num_epochs):
            # Training on cot dataset
            for batch in self.cot_dataloader:

                self.optimizer.zero_grad()
                loss = self.train_step(batch)
                # Training on correct dataset
                if torch.rand(1).item() < self.lambda_param:
                    correct_batch = next(iter(self.correct_dataloader))
                    loss += self.train_step(correct_batch)

                # Log the loss to wandb
                wandb.log({"train_loss": loss.item()})

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()

            # Validation
            with torch.inference_mode():
                for batch in self.validation_dataloader:
                    accuracy = self.validation_step(batch)
                    total_accuracy += accuracy.item()

            avg_loss = total_loss / len(self.cot_dataloader)
            avg_accuracy = total_accuracy / len(self.validation_dataloader)
            print(
                f"Epoch {epoch+1}/{self.num_epochs}, Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}"
            )
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "average_loss": avg_loss,
                    "average_accuracy": avg_accuracy,
                }
            )

            # Perform early stopping
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_epoch = epoch + 1
            else:
                if epoch - self.best_epoch > self.patience:
                    print("Early stopping triggered")
                    break

        self.model.save_pretrained("./models/genrm_cot_final")


def main():

    # Initialize and train the model
    trainer = GenRMCoTTrainer(GenRMCoTTrainerConfig)

    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
