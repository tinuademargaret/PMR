# %%
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os
from dotenv import load_dotenv

load_dotenv()


# %%
class ImplicitBiasDataset(Dataset):
    def __init__(
        self, data, tokenizer, max_length=512, dataset_type="cot", mode="train"
    ):
        """
        Initialize the GenRM-CoT dataset.

        Args:
            data (pd.DataFrame): DataFrame containing the dataset.
            tokenizer (AutoTokenizer): Tokenizer for encoding the text.
            max_length (int): Maximum sequence length for tokenization.
            dataset_type (str): Type of dataset ('cot' or 'correct').
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_type = dataset_type
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            dict: Encoded input and output tensors.
        """
        row = self.data.iloc[idx]

        if self.dataset_type == "cot":
            x = row["task"]
            y = row["biased_response"] if idx % 2 == 0 else row["preferred_response"]
            I_cot = "Verify the unbiasedness of the given solution."
            v_cot = row["reasoning"]
            I = "Is the answer unbiased?"
            answer = "No" if idx % 2 == 0 else "Yes"

            input_text = f"{x}\n{y}\n{I_cot}"
            output_text = f"{v_cot}\n{I}\n{answer}"

            if self.mode == "test":
                return {
                    "x": x,
                    "y": y,
                    "I_cot": I_cot,
                    "I": I,
                    "answer": answer,
                }
        else:  # 'correct' dataset
            x = row["task"]
            y = row["preferred_response"]
            input_text = x
            output_text = y

        input_encoding = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        output_encoding = self.tokenizer.encode_plus(
            output_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": output_encoding["input_ids"].squeeze(),
        }


def prepare_train_datasets(data_path, tokenizer):
    """
    Prepare D_cot and D_correct datasets from the decision_data.csv file.

    Args:
        data_path (str): Path to the decision_data.csv file.
        tokenizer (AutoTokenizer): Tokenizer for encoding the text.

    Returns:
        tuple: D_cot and D_correct datasets.
    """
    df = pd.read_csv(data_path)

    # Create D_cot dataset
    d_cot = ImplicitBiasDataset(df, tokenizer, dataset_type="cot")

    # Create D_correct dataset
    d_correct = ImplicitBiasDataset(df, tokenizer, dataset_type="correct")

    return d_cot, d_correct


def prepare_test_datasets(data_path, tokenizer):
    """
    Prepare D_cot and D_correct datasets from the decision_data.csv file.

    Args:
        data_path (str): Path to the decision_data.csv file.
        tokenizer (AutoTokenizer): Tokenizer for encoding the text.
    """
    df = pd.read_csv(data_path)

    # Create D_cot dataset
    d_cot = ImplicitBiasDataset(df, tokenizer, dataset_type="cot", mode="test")

    return d_cot


def get_dataloader(dataset, batch_size=32, shuffle=True):
    """
    Create a DataLoader for the given dataset.

    Args:
        dataset (GenRMCoTDataset): Dataset to create a DataLoader for.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
