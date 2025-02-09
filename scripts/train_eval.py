import argparse
import json
import os
import torch
import glob
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from transformers import AutoModel, AutoTokenizer, MT5EncoderModel
from peft import PeftModel
from datasets import load_dataset, concatenate_datasets

# Use the new modular alignment model
from pool.model import LangBridgeModular


###############################################
# Collate function that formats a batch.
# Expects each sample to have "text1" and "text2" keys.
###############################################
def collate_fn(batch):
    return {
        'text1': [sample['text1'] for sample in batch],
        'text2': [sample['text2'] for sample in batch]
    }


###############################################
# Main training and evaluation routine.
###############################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help="Path to JSON config file")
    args = parser.parse_args()

    # Load JSON configuration.
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Initialize WandB logging.
    wandb.init(project=config.get("wandb_project", "LangBridgeModular"), config=config)

    # Set device to GPU if available, else CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###############################################
    # 1. Load the encoder model and apply PEFT.
    ###############################################
    encoder_variant = config.get("encoder_variant", "AMS")
    loadpath = f"../trained_models/mST5-{encoder_variant}-final-true"
    encoder_model = MT5EncoderModel.from_pretrained(
        "../mST5-saved-2",
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    )
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-xxl", use_fast=False)
    encoder_model = PeftModel.from_pretrained(encoder_model, loadpath)
    encoder_model = encoder_model.merge_and_unload()
    encoder_model.to(device)

    ###############################################
    # 2. Load the decoder model.
    ###############################################
    decoder_model = AutoModel.from_pretrained(config["decoder_model_name_or_path"]).to(device)

    ###############################################
    # 3. Select and combine datasets based on the decoder model.
    # Different tasks (e.g. math reasoning, code completion, etc.) use distinct datasets.
    ###############################################
    decoder_name = config["decoder_model_name_or_path"].lower()
    train_datasets = []
    eval_datasets = []
    if config.get(custom_datasets):
        train_datasets.append(load_dataset("DKYoon/proofpile2-200k", split="train"))
        eval_datasets.append(load_dataset("juletxara/mgsm", split="train"))

    elif "metamath" in decoder_name or "llemma" in decoder_name:
        # Math reasoning: combine multiple datasets.
        train_datasets.append(load_dataset(config.get("metamath_train_dataset", "kaist-ai/metamathqa"), split="train"))
        train_datasets.append(load_dataset("juletxara/mgsm", split="train"))
        train_datasets.append(load_dataset("Mathoctopus/MSVAMP", split="train"))
        eval_datasets.append(load_dataset(config.get("metamath_eval_dataset", "kaist-ai/mgsm"), split="train"))
        eval_datasets.append(load_dataset("juletxara/mgsm", split="train"))
        eval_datasets.append(load_dataset("Mathoctopus/MSVAMP", split="train"))
        task = "math_reasoning"
    elif "codellama" in decoder_name:
        # Code completion: use HumanEval and HumanEval-MT datasets.
        train_datasets.append(load_dataset(config.get("codellama_train_dataset", "kaist-ai/starcoder_python"), split="train"))
        train_datasets.append(load_dataset("openai/openai_humaneval", split="train"))
        mt_files = glob.glob("../humaneval/humaneval_*.json")
        for f in mt_files:
            train_datasets.append(load_dataset("json", data_files=f, split="train"))
        eval_datasets.append(load_dataset(config.get("codellama_eval_dataset", "kaist-ai/humaneval"), split="train"))
        eval_datasets.append(load_dataset("openai/openai_humaneval", split="train"))
        for f in mt_files:
            eval_datasets.append(load_dataset("json", data_files=f, split="train"))
        task = "code_completion"
    elif "orca2" in decoder_name:
        # Logical/commonsense reasoning: combine multiple datasets.
        train_datasets.append(load_dataset(config.get("orca_train_dataset", "kaist-ai/openorca"), split="train"))
        train_datasets.append(load_dataset("cambridgeltl/xcopa", split="train"))
        train_datasets.append(load_dataset("lukaemon/bbh", split="train"))
        train_datasets.append(load_dataset(config.get("bbh_bn_train_dataset", "lukaemon/bbh_bn"), split="train"))
        eval_datasets.append(load_dataset(config.get("orca_eval_dataset", "kaist-ai/bbh"), split="train"))
        eval_datasets.append(load_dataset("cambridgeltl/xcopa", split="train"))
        eval_datasets.append(load_dataset("lukaemon/bbh", split="train"))
        eval_datasets.append(load_dataset(config.get("bbh_bn_eval_dataset", "lukaemon/bbh_bn"), split="train"))
        task = "logical_reasoning"
    else:
        # Default datasets.
        train_datasets.append(load_dataset(config.get("hf_train_dataset", "default/train"), split="train"))
        eval_datasets.append(load_dataset(config.get("hf_eval_dataset", "default/eval"), split="train"))
        task = "default"

    # Concatenate multiple datasets if necessary.
    train_dataset = concatenate_datasets(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    eval_dataset = concatenate_datasets(eval_datasets) if len(eval_datasets) > 1 else eval_datasets[0]

    # Update WandB config with dataset details.
    wandb.config.update({
        "hf_train_datasets": [ds.info.builder_name if hasattr(ds, "info") and ds.info.builder_name else "local"
                                for ds in train_datasets],
        "hf_eval_datasets": [ds.info.builder_name if hasattr(ds, "info") and ds.info.builder_name else "local"
                               for ds in eval_datasets],
        "task": task
    })

    # Remove any unneeded columns, keeping only "text1" and "text2".
    for ds in [train_dataset, eval_dataset]:
        cols_to_remove = [col for col in ds.column_names if col not in ["text1", "text2"]]
        if cols_to_remove:
            ds = ds.remove_columns(cols_to_remove)

    # Create data loaders.
    train_loader = DataLoader(train_dataset,
                              batch_size=config.get("train_batch_size", 8),
                              shuffle=True,
                              collate_fn=collate_fn)
    eval_loader  = DataLoader(eval_dataset,
                              batch_size=config.get("eval_batch_size", 8),
                              shuffle=False,
                              collate_fn=collate_fn)

    ###############################################
    # 4. Instantiate the LangBridgeModular model.
    # Pass in aggregator and alignment types via the config.
    ###############################################
    model = LangBridgeModular(
        encoder_model=encoder_model,
        decoder_model=decoder_model,
        tokenizer=tokenizer,
        aggregator_type=config.get("aggregator_type", "max_pool"),
        alignment_type=config.get("alignment_type", "LinearWithAddedEos"),
        fine_tune_encoder=config.get("fine_tune_encoder", True),
        max_sentence_length=config.get("max_sentence_length", 32),
        prompt_length=config.get("prompt_length", 10)
    ).to(device)

    # Set up optimizer and loss function.
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=config.get("learning_rate", 1e-4))
    cosine_loss = nn.CosineEmbeddingLoss()

    num_epochs = config.get("num_epochs", 3)
    global_step = 0
    best_eval_loss = float("inf")
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    ###############################################
    # 5. Main training loop.
    # Perform a forward pass on paired texts and compute cosine similarity loss.
    ###############################################
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            # Process each pair of texts.
            outputs1, soft_prompt1 = model(batch['text1'])
            outputs2, soft_prompt2 = model(batch['text2'])
            # Compute the average over the sequence dimension.
            prompt_avg1 = soft_prompt1.mean(dim=1)
            prompt_avg2 = soft_prompt2.mean(dim=1)
            # Target tensor for cosine similarity (all ones).
            target = torch.ones(prompt_avg1.size(0), device=device)
            loss = cosine_loss(prompt_avg1, prompt_avg2, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            global_step += 1
            wandb.log({"train_loss": loss.item(), "global_step": global_step})
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_epoch_loss}")

        # Save checkpoint after each epoch.
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "epoch": epoch + 1
        }, checkpoint_path)

        ###############################################
        # 6. Evaluation phase.
        # Evaluate the model on the eval_loader and compute average loss.
        ###############################################
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch in eval_loader:
                outputs1, soft_prompt1 = model(batch['text1'])
                outputs2, soft_prompt2 = model(batch['text2'])
                prompt_avg1 = soft_prompt1.mean(dim=1)
                prompt_avg2 = soft_prompt2.mean(dim=1)
                target = torch.ones(prompt_avg1.size(0), device=device)
                loss = cosine_loss(prompt_avg1, prompt_avg2, target)
                eval_loss += loss.item()
        avg_eval_loss = eval_loss / len(eval_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Eval Loss: {avg_eval_loss}")
        wandb.log({"eval_loss": avg_eval_loss, "epoch": epoch+1})

        # Save best checkpoint based on eval loss.
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "epoch": epoch + 1
            }, best_checkpoint_path)
    
    ###############################################
    # 7. Save final evaluation metrics.
    ###############################################
    final_metrics = {
        "best_eval_loss": best_eval_loss,
        "num_epochs": num_epochs,
        "global_steps": global_step,
        "task": task
    }
    eval_output_path = config.get("eval_output_path", "final_eval_metrics.json")
    with open(eval_output_path, "w") as f:
        json.dump(final_metrics, f, indent=4)
    print(f"Final evaluation metrics saved to {eval_output_path}")

if __name__ == "__main__":
    main()