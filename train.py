"""
This script is the main application for training a Transformer-based machine translation model.
It uses Hydra for configuration management, performs dataset loading, model initialization,
training, evaluation, and sends an email notification upon completion.

Author: yumemonzo@gmail.com
Date: 2025-03-02
"""

import os
import logging
import pickle
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from data import WMT14EnDeDataset, get_loaders
from model import Transformer
from trainer import Trainer
from utils import seed_everything, count_model_parameters, send_email


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    """
    Main function to set up data, model, training, and evaluation for the machine translation task.
    
    Args:
        cfg (DictConfig): Configuration object with training, data, model, and email parameters.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    seed_everything(cfg['seed'])

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-ko")
    opus100_en_ko_dataset = WMT14EnDeDataset(tokenizer, cfg['data']['max_length'])

    data_dir: str = os.path.join(cfg['data']['data_dir'], str(cfg['data']['max_length']))
    if os.path.exists(data_dir):
        tokenized_datasets = opus100_en_ko_dataset.load_tokenized_dataset(data_dir)
    else:
        os.makedirs(data_dir)
        tokenized_datasets = opus100_en_ko_dataset.create_tokenized_dataset(data_dir)

    opus100_en_ko_dataset.show_sample(tokenized_datasets)

    data_collator = opus100_en_ko_dataset.get_data_collator()
    train_loader, valid_loader, test_loader = get_loaders(
        tokenized_datasets["train"],
        tokenized_datasets["validation"],
        tokenized_datasets["test"],
        data_collator,
        cfg['data']['batch_size'],
        cfg['data']['num_workers']
    )
    logger.info(f"Train Loader: {len(train_loader)}")
    logger.info(f"Valid Loader: {len(valid_loader)}")
    logger.info(f"Test Loader: {len(test_loader)}")

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    model = Transformer(
        cfg['model']['num_layers'],
        cfg['model']['d_model'],
        cfg['model']['num_heads'],
        cfg['model']['d_ff'],
        tokenizer.vocab_size,
        tokenizer.vocab_size,
        cfg['data']['max_length'],
        cfg['model']['dropout_p']
    )
    model.to(device)
    logger.info(f"Model Parameters: {count_model_parameters(model):,}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg['train']['lr'],
        betas=(0.9, 0.98),
        eps=1e-3
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100).to(device)

    num_training_steps: int = cfg['train']['epochs'] * len(train_loader)
    num_warmup_steps: int = int(num_training_steps * cfg['train']['warmup_ratio'])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    output_dir: str = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        criterion,
        tokenizer,
        cfg['data']['max_length'],
        cfg['train']['patience'],
        device,
        logger,
        output_dir
    )

    weight_path: str = os.path.join(output_dir, "best_model.pth")
    train_losses, valid_losses, bleu_score, all_references, all_hypotheses = trainer.training(
        cfg['train']['epochs'],
        train_loader,
        valid_loader,
        test_loader,
        weight_path
    )

    pickle_path: str = os.path.join(output_dir, "training_results.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump({
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "bleu_score": bleu_score,
            "all_references": all_references,
            "all_hypotheses": all_hypotheses,
        }, f)

    subject: str = "Training Completed"
    body: str = f"Training job has completed successfully.\nFinal BLEU Score: {bleu_score}"
    send_email(subject, body, cfg['email']['to'], cfg['email']['from'], cfg['email']['password'])


if __name__ == "__main__":
    my_app()
