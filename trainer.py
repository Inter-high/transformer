"""
This module defines the Trainer class to manage the training, validation, and testing of a
machine translation model. It includes functionalities for early stopping, BLEU score evaluation,
and TensorBoard logging.

Author: yumemonzo@gmail.com
Date: 2025-03-02
"""

import time
import datetime
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import generate, calculate_bleu_score
from typing import Tuple, List, Any


class Trainer:
    """
    Trainer class to handle the training loop, validation, testing, early stopping, and logging.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Any,
                 criterion: torch.nn.Module,
                 tokenizer: Any,
                 max_length: int,
                 patience: int,
                 device: torch.device,
                 logger: Any,
                 log_dir: str) -> None:
        """
        Initialize the Trainer with the model, optimizer, scheduler, loss function, tokenizer,
        maximum sequence length, early stopping patience, device, logger, and TensorBoard log directory.

        Args:
            model (torch.nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer for parameter updates.
            scheduler (Any): Learning rate scheduler.
            criterion (torch.nn.Module): Loss function.
            tokenizer (Any): Tokenizer for encoding and decoding sequences.
            max_length (int): Maximum sequence length for generation.
            patience (int): Number of epochs to wait for improvement before stopping.
            device (torch.device): Device for computations.
            logger (Any): Logger to record training progress.
            log_dir (str): Directory path for TensorBoard logs.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.lowest_loss = float("inf")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.logger = logger
        self.writer = SummaryWriter(log_dir=log_dir)
        self.patience = patience
        self.no_improve_count = 0

    def train(self, train_loader: torch.utils.data.DataLoader) -> float:
        """
        Perform one epoch of training on the training dataset.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Training", leave=True)

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Prepare target input and labels for teacher forcing (shifted)
            tgt_input = labels[:, :-1]
            tgt_labels = labels[:, 1:]

            self.optimizer.zero_grad()
            y_hat = self.model(input_ids, tgt_input, src_mask=attention_mask)
            y_hat = y_hat.permute(0, 2, 1)  # Rearrange dimensions for loss computation
            loss = self.criterion(y_hat, tgt_labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def valid(self, valid_loader: torch.utils.data.DataLoader) -> float:
        """
        Evaluate the model on the validation dataset.

        Args:
            valid_loader (torch.utils.data.DataLoader): DataLoader for validation data.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        progress_bar = tqdm(valid_loader, desc="Validating", leave=True)

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Prepare target input and labels for teacher forcing (shifted)
                tgt_input = labels[:, :-1]
                tgt_labels = labels[:, 1:]

                y_hat = self.model(input_ids, tgt_input, src_mask=attention_mask)
                y_hat = y_hat.permute(0, 2, 1)
                loss = self.criterion(y_hat, tgt_labels)
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(valid_loader)
        return avg_loss

    def test(self, test_loader: torch.utils.data.DataLoader) -> Tuple[List[List[List[str]]], List[List[str]]]:
        """
        Evaluate the model on the test dataset and generate predictions for BLEU score calculation.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader for test data.

        Returns:
            Tuple[List[List[List[str]]], List[List[str]]]: A tuple containing lists of references and hypotheses.
        """
        self.model.eval()
        all_references = []
        all_hypotheses = []
        progress_bar = tqdm(test_loader, desc="Testing", leave=True)

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]

                generated_ids = generate(self.model, input_ids, attention_mask, self.tokenizer, self.device, self.max_length)

                for ref_ids, gen_ids in zip(labels, generated_ids):
                    reference = self.tokenizer.decode(ref_ids, skip_special_tokens=True).split()
                    hypothesis = self.tokenizer.decode(gen_ids, skip_special_tokens=True).split()
                    all_references.append([reference])
                    all_hypotheses.append(hypothesis)

        return all_references, all_hypotheses

    def training(self,
                 epochs: int,
                 train_loader: torch.utils.data.DataLoader,
                 valid_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
                 weight_path: str) -> Tuple[List[float], List[float], float, List[List[List[str]]], List[List[str]]]:
        """
        Run the full training process across multiple epochs, including training, validation,
        testing, early stopping, and logging.

        Args:
            epochs (int): Total number of epochs for training.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            valid_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            test_loader (torch.utils.data.DataLoader): DataLoader for test data.
            weight_path (str): File path to save the best model weights.

        Returns:
            Tuple[List[float], List[float], float, List[List[List[str]]], List[List[str]]]:
                A tuple containing the training loss history, validation loss history, final BLEU score,
                list of reference translations, and list of hypothesis translations.
        """
        start_time = time.time()  # Record training start time
        train_losses = []
        valid_losses = []

        for epoch in range(1, epochs + 1):
            train_loss = self.train(train_loader)
            train_losses.append(train_loss)
            loss_dict = {"train": train_loss}

            # Validate every 5 epochs
            if epoch % 5 == 0:
                valid_loss = self.valid(valid_loader)
                valid_losses.append(valid_loss)
                loss_dict["valid"] = valid_loss

                self.logger.info(f"Epoch: {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")

                # Early stopping: Save the model if validation loss improves
                if valid_loss < self.lowest_loss:
                    self.lowest_loss = valid_loss
                    self.no_improve_count = 0  # Reset counter on improvement
                    torch.save(self.model.state_dict(), weight_path)
                    self.logger.info(f"New lowest valid loss: {valid_loss:.4f}. Model weights saved to {weight_path}")
                else:
                    self.no_improve_count += 1
                    self.logger.info(f"No improvement count: {self.no_improve_count}/{self.patience}")
                    if self.no_improve_count >= self.patience:
                        self.logger.info("Early stopping triggered.")
                        break

                # Run testing for BLEU score evaluation even if early stopping is triggered
                all_references, all_hypotheses = self.test(test_loader)
                bleu_score = calculate_bleu_score(all_references, all_hypotheses)
                self.logger.info(f"Epoch: {epoch}/{epochs} | Bleu Score: {bleu_score:.8f}")

            self.writer.add_scalars("Loss", loss_dict, epoch)
            self.scheduler.step()

            # Estimate and log remaining training time
            elapsed_time = time.time() - start_time
            avg_epoch_time = elapsed_time / epoch
            remaining_epochs = epochs - epoch
            est_remaining_time = avg_epoch_time * remaining_epochs
            formatted_remaining_time = str(datetime.timedelta(seconds=int(est_remaining_time)))
            self.logger.info(f"Epoch {epoch}/{epochs} completed. Estimated remaining time: {formatted_remaining_time}")

        total_elapsed_time = time.time() - start_time
        formatted_total_time = str(datetime.timedelta(seconds=int(total_elapsed_time)))
        self.logger.info(f"Total training time: {formatted_total_time}")

        return train_losses, valid_losses, bleu_score, all_references, all_hypotheses
