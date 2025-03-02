"""
This module defines a dataset class for handling the WMT14 English-German translation dataset,
including functionality for tokenizing, saving, and loading the dataset. It also provides a function
to create DataLoader instances for training, validation, and testing.

Author: yumemonzo@gmail.com
Date: 2025-03-02
"""

from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizer
from typing import Any, Dict, List, Optional, Tuple


class WMT14EnDeDataset:
    """
    A dataset class for handling WMT14 English-German translation data.
    
    Attributes:
        tokenizer (PreTrainedTokenizer): The tokenizer used for processing text.
        max_length (int): The maximum sequence length for tokenization.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int) -> None:
        """
        Initializes the dataset with a tokenizer and maximum sequence length.
        
        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer for encoding text.
            max_length (int): Maximum token length for each input sequence.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize_function(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenizes the English and German texts in the dataset example.
        
        Args:
            example (Dict[str, Any]): A dictionary containing a 'translation' key, which includes a list
                of dictionaries with 'en' and 'de' keys for English and German texts respectively.
        
        Returns:
            Dict[str, Any]: A dictionary with tokenized inputs and labels.
        """
        # Extract English and German texts from the translation field
        en_texts: List[str] = [ex["en"] for ex in example["translation"]]
        de_texts: List[str] = [ex["de"] for ex in example["translation"]]

        # Tokenize the English texts as model inputs
        model_inputs: Dict[str, Any] = self.tokenizer(en_texts, max_length=self.max_length, truncation=True)
        
        # Tokenize the German texts as target sequences
        with self.tokenizer.as_target_tokenizer():
            # Adjust target maximum length if eos_token_id is available
            target_max_length: int = self.max_length - 1 if (self.tokenizer.eos_token_id is not None) else self.max_length
            labels: Dict[str, Any] = self.tokenizer(de_texts, max_length=target_max_length, truncation=True)

        # Prepend the EOS token to the beginning of labels if available
        if self.tokenizer.eos_token_id is not None:
            labels["labels"] = [[self.tokenizer.eos_token_id] + label for label in labels["input_ids"]]
        
        model_inputs["labels"] = labels["labels"]

        return model_inputs

    def get_data_collator(self) -> DataCollatorForSeq2Seq:
        """
        Returns a DataCollator for sequence-to-sequence tasks.
        
        Returns:
            DataCollatorForSeq2Seq: A collator that batches tokenized data for seq2seq tasks.
        """
        return DataCollatorForSeq2Seq(self.tokenizer, model=None, label_pad_token_id=self.tokenizer.pad_token_id)

    def create_tokenized_dataset(self, data_dir: Optional[str] = None) -> Any:
        """
        Creates a tokenized dataset from the WMT14 English-German dataset.
        
        Args:
            data_dir (Optional[str]): Directory path to save the tokenized dataset. If provided, the dataset
                                      will be saved to disk.
        
        Returns:
            Any: The tokenized dataset.
        """
        # Load the WMT14 dataset for English-German translation (using the 'de-en' configuration)
        dataset: Any = load_dataset("wmt14", "de-en")
        # Apply tokenization to the entire dataset
        tokenized_datasets: Any = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )

        # Save the tokenized dataset to disk if a directory is provided
        if data_dir:
            tokenized_datasets.save_to_disk(data_dir)

        return tokenized_datasets

    def load_tokenized_dataset(self, data_dir: Optional[str] = None) -> Any:
        """
        Loads a tokenized dataset from disk.
        
        Args:
            data_dir (Optional[str]): Directory path from which to load the tokenized dataset.
        
        Returns:
            Any: The loaded tokenized dataset.
        """
        return load_from_disk(data_dir)

    def show_sample(self, tokenized_dataset: Any, idx: int = 1) -> None:
        """
        Displays a sample input and label from the tokenized dataset.
        
        Args:
            tokenized_dataset (Any): The tokenized dataset containing the 'train' split.
            idx (int, optional): The index of the sample to display. Defaults to 1.
        """
        # Retrieve the sample from the 'train' split at the given index
        sample = tokenized_dataset["train"][idx]

        # Decode input and label sequences into human-readable strings
        input_text: str = self.tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
        label_text: str = self.tokenizer.decode(sample["labels"], skip_special_tokens=False)

        print(f"Sample input sentence: {input_text}")
        print(f"Sample label sentence: {label_text}")


def get_loaders(train_dataset: Any, valid_dataset: Any, test_dataset: Any,
                data_collator: Any, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoader instances for training, validation, and testing datasets.
    
    Args:
        train_dataset (Any): The training dataset.
        valid_dataset (Any): The validation dataset.
        test_dataset (Any): The testing dataset.
        data_collator (Any): Function to collate data into batches.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: DataLoaders for training, validation, and testing.
    """
    # Create DataLoader for training data with shuffling enabled
    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=data_collator
    )
    
    # Create DataLoader for validation data without shuffling
    valid_loader: DataLoader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_collator
    )
    
    # Create DataLoader for testing data with a reduced batch size
    test_loader: DataLoader = DataLoader(
        test_dataset,
        batch_size=(batch_size // 2),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_collator
    )
    
    return train_loader, valid_loader, test_loader
