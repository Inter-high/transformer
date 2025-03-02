"""
This module provides utility functions for reproducibility, model evaluation, loss visualization,
sequence generation, BLEU score calculation, and email sending functionality for NLP tasks.

Author: yumemonzo@gmail.com
Date: 2025-03-02
"""

import random
import torch
import matplotlib.pyplot as plt
from typing import List, Any
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for Python and PyTorch to ensure reproducibility.
    
    Args:
        seed (int): Seed value to use (default is 42).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_model_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.
    
    Args:
        model (torch.nn.Module): The model whose parameters are to be counted.
    
    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_compare_loss(data: List[List[float]], labels: List[str], save_path: str, title: str = 'Loss Comparison') -> None:
    """
    Plot multiple loss curves for different experiments and save the figure.
    
    Args:
        data (List[List[float]]): List of loss values for each experiment.
        labels (List[str]): List of labels for each experiment.
        save_path (str): File path to save the plot.
        title (str): Title of the plot (default is 'Loss Comparison').
    
    Raises:
        ValueError: If the number of data series does not match the number of labels.
    """
    if len(data) != len(labels):
        raise ValueError("The lengths of data and labels must be equal.")
    
    plt.figure(figsize=(10, 6))
    for i, losses in enumerate(data):
        epochs = range(1, len(losses) + 1)
        plt.plot(epochs, losses, label=labels[i])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.clf()


def generate(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer: Any, device: torch.device, max_length: int) -> torch.Tensor:
    """
    Generate sequences from the model using the provided input and attention mask.
    
    Args:
        model (torch.nn.Module): The model used for generation.
        input_ids (torch.Tensor): Tensor of input token IDs.
        attention_mask (torch.Tensor): Tensor indicating which tokens to attend to.
        tokenizer (Any): Tokenizer to decode the generated tokens.
        device (torch.device): Device on which the model is running.
        max_length (int): Maximum length for the generated sequence.
    
    Returns:
        torch.Tensor: Generated sequence of token IDs.
    """
    model.eval()
    bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    batch_size = input_ids.size(0)
    generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_ids, generated, src_mask=attention_mask)
            next_token_logits = output[:, -1, :]
            next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tokens], dim=1)
            if (next_tokens == tokenizer.eos_token_id).all():
                break
    return generated


def calculate_bleu_score(all_references: List[List[List[str]]], all_hypotheses: List[List[str]]) -> float:
    """
    Calculate the BLEU score for a set of reference and hypothesis translations.
    
    Args:
        all_references (List[List[List[str]]]): List of reference translations (each reference is a list of tokens).
        all_hypotheses (List[List[str]]): List of hypothesis translations (each hypothesis is a list of tokens).
    
    Returns:
        float: Computed BLEU score.
    """
    smooth_fn = SmoothingFunction().method1
    bleu_score = corpus_bleu(all_references, all_hypotheses, smoothing_function=smooth_fn)
    return bleu_score


def send_email(subject: str, body: str, to_email: str, from_email: str, password: str,
               smtp_server: str = "smtp.gmail.com", port: int = 587) -> None:
    """
    Send an email with the specified subject and body using SMTP.
    
    Args:
        subject (str): Subject line of the email.
        body (str): Body content of the email.
        to_email (str): Recipient's email address.
        from_email (str): Sender's email address.
        password (str): Password for the sender's email account.
        smtp_server (str): SMTP server address (default is "smtp.gmail.com").
        port (int): Port number for the SMTP server (default is 587).
    
    Returns:
        None
    """
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    
    try:
        server = smtplib.SMTP(smtp_server, port)
        server.starttls()  # Initiate TLS for security
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")
