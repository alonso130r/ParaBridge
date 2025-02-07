import argparse
import json
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from transformers import AutoModel, AutoTokenizer, MT5EncoderModel
from peft import PeftModel
from datasets import load_dataset
from model import LangBridgeWithLSTM

# A collate function that expects each sample to have keys "text1" and "text2"
def collate_fn(batch):
    text1_list = [sample['text1'] for sample in batch]
    text2_list = [sample['text2'] for sample in batch]
    return {'text1': text1_list, 'text2': text2_list}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help="Path to JSON config file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Initialize WandB logging
    wandb.init(project=config.get("wandb_project", "LangBridgeWithLSTM"), config=config)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #########################################
    # 1. Load Encoder (from a fixed load path)
    #########################################
    # The config should provide an "encoder_variant" key with value either "AMS" or "barlow"
    encoder_variant = config.get("encoder_variant", "AMS")
    # Build the load path (for example: "../trained_models/mST5-AMS-final-true")
    loadpath = f"../trained_models/mST5-{encoder_variant}-final-true"

    # Load the base MT5 encoder model (always from the same directory)
    # (Note: device_map is passed as a dictionary mapping; adjust as needed)
    encoder_model = MT5EncoderModel.from_pretrained(
        "../mST5-saved-2",
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    )
    # Load the corresponding tokenizer (we use google/mt5-xxl here)
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-xxl", use_fast=False)
    # Wrap with PeftModel using the provided loadpath and then merge and unload the adapter parameters
    encoder_model = PeftModel.from_pretrained(encoder_model, loadpath)
    encoder_model = encoder_model.merge_and_unload()
    encoder_model.to(device)

    #########################################
    # 2. Load Decoder Model (as usual)
    #########################################
    decoder_model = AutoModel.from_pretrained(config["decoder_model_name_or_path"]).to(device)

    #########################################
    # 3. Dataset selection based on the decoder model
    #########################################
    decoder_name = config["decoder_model_name_or_path"].lower()
    if "metamath" in decoder_name:
        hf_train_dataset_id = config.get("metamath_train_dataset", "kaist-ai/metamathqa")
        hf_eval_dataset_id  = config.get("metamath_eval_dataset", "kaist-ai/mgsm")
        task = "math_reasoning"
    elif "llemma" in decoder_name:
        hf_train_dataset_id = config.get("llemma_train_dataset", "kaist-ai/proof-pile-2")
        hf_eval_dataset_id  = config.get("llemma_eval_dataset", "kaist-ai/mgsm")
        task = "math_reasoning"
    elif "codellama" in decoder_name:
        hf_train_dataset_id = config.get("codellama_train_dataset", "kaist-ai/starcoder_python")
        hf_eval_dataset_id  = config.get("codellama_eval_dataset", "kaist-ai/humaneval")
        task = "code_completion"
    elif "orca2" in decoder_name:
        hf_train_dataset_id = config.get("orca_train_dataset", "kaist-ai/openorca")
        hf_eval_dataset_id  = config.get("orca_eval_dataset", "kaist-ai/bbh")
        task = "logical_reasoning"
    else:
        hf_train_dataset_id = config.get("hf_train_dataset", "default/train")
        hf_eval_dataset_id  = config.get("hf_eval_dataset", "default/eval")
        task = "default"

    wandb.config.update({"hf_train_dataset_id": hf_train_dataset_id,
                           "hf_eval_dataset_id": hf_eval_dataset_id,
                           "task": task})

    #########################################
    # 4. Load datasets directly from Hugging Face (assumed to be parquet files)
    #########################################
    print(f"Loading training dataset from HF dataset: {hf_train_dataset_id}")
    train_dataset = load_dataset(hf_train_dataset_id, split="train")
    print(f"Loading evaluation dataset from HF dataset: {hf_eval_dataset_id}")
    eval_dataset  = load_dataset(hf_eval_dataset_id, split="train")
    # Optionally, remove any columns other than "text1" and "text2"
    for ds in [train_dataset, eval_dataset]:
        cols_to_remove = [col for col in ds.column_names if col not in ["text1", "text2"]]
        if cols_to_remove:
            ds = ds.remove_columns(cols_to_remove)

    train_loader = DataLoader(train_dataset,
                              batch_size=config.get("train_batch_size", 8),
                              shuffle=True,
                              collate_fn=collate_fn)
    eval_loader  = DataLoader(eval_dataset,
                              batch_size=config.get("eval_batch_size", 8),
                              shuffle=False,
                              collate_fn=collate_fn)

    #########################################
    # 5. Instantiate LangBridge model
    #########################################
    model = LangBridgeWithLSTM(
        encoder_model=encoder_model,
        decoder_model=decoder_model,
        tokenizer=tokenizer,
        fine_tune_encoder=config.get("fine_tune_encoder", True),
        lstm_num_layers=config.get("lstm_num_layers", 1),
        max_sentence_length=config.get("max_sentence_length", 32),
        prompt_length=config.get("prompt_length", 10)
    ).to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=config.get("learning_rate", 1e-4))
    cosine_loss = nn.CosineEmbeddingLoss()

    num_epochs = config.get("num_epochs", 3)
    global_step = 0
    best_eval_loss = float("inf")
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    #########################################
    # 6. Training Loop
    #########################################
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            # Process paired texts from the batch
            outputs1, soft_prompt1 = model(batch['text1'])
            outputs2, soft_prompt2 = model(batch['text2'])
            # Average the soft prompt tokens over the sequence dimension
            prompt_avg1 = soft_prompt1.mean(dim=1)
            prompt_avg2 = soft_prompt2.mean(dim=1)
            target = torch.ones(prompt_avg1.size(0), device=device)
            loss = cosine_loss(prompt_avg1, prompt_avg2, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            global_step += 1
            wandb.log({"train_loss": loss.item(), "global_step": global_step})
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_epoch_loss}")
        
        # Save a checkpoint at the end of each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "epoch": epoch + 1
        }, checkpoint_path)

        #########################################
        # 7. Evaluation Phase
        #########################################
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
        
        # Save best model checkpoint if evaluation loss improves
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "epoch": epoch + 1
            }, best_checkpoint_path)
    
    #########################################
    # 8. Save Final Evaluation Metrics
    #########################################
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
