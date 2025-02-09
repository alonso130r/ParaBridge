import argparse
import json
import os
import glob
import torch
from accelerate import Accelerator
import wandb

# Transformers & PEFT
from transformers import AutoModel, AutoTokenizer, MT5EncoderModel
from peft import PeftModel

# EleutherAI LM Evaluation Harness imports
from lm_eval.evaluate import evaluate
from lm_eval.models.model import Model

# For loading datasets from Hugging Face
from datasets import load_dataset

# Use the new modular alignment model
from model import LangBridgeModular


###############################################
# Custom task class for Hugging Face datasets.
# This class assumes that each dataset example is a dictionary
# with keys "prompt" and "answer".
###############################################
class HFTask:
    def __init__(self, hf_dataset, name, system_prompt=None):
        self.hf_dataset = hf_dataset
        self.name = name
        self.system_prompt = system_prompt

    def has_validation_docs(self):
        return True

    def validation_docs(self):
        return self.hf_dataset

    def doc_to_text(self, doc):
        # Prepend system prompt if provided.
        if self.system_prompt:
            return f"{self.system_prompt}\n{doc['prompt']}"
        return doc["prompt"]

    def doc_to_target(self, doc):
        # Return the expected answer stripped of whitespace.
        return doc["answer"].strip()

    def postprocess(self, generated_text):
        # Remove any system prompt prefix from the generated text.
        if self.system_prompt and generated_text.startswith(self.system_prompt):
            return generated_text[len(self.system_prompt):].strip()
        return generated_text.strip()

    def aggregation(self):
        # Define an accuracy metric.
        return {"accuracy": lambda items: sum(items) / len(items) if items else 0.0}

    def higher_is_better(self):
        return {"accuracy": True}

    def fewshot_description(self):
        return "0-shot evaluation"


###############################################
# Wrapper class to generate outputs using the modular model.
###############################################
class LangBridgeModelWrapper:
    def __init__(self, model, decoder_model, tokenizer, device):
        self.model = model
        self.decoder_model = decoder_model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, prompt, max_length=128, **kwargs):
        # Switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            # The model expects a list of prompts and outputs soft prompt embeddings.
            _, soft_prompt = self.model([prompt])
            # Use the decoder model to generate token ids based on soft prompts.
            generated_ids = self.decoder_model.generate(
                inputs_embeds=soft_prompt,
                max_length=max_length,
                **kwargs
            )
            # Decode generated ids back to text.
            generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()


###############################################
# Wrapper class to conform our model interface to the LM Evaluation Harness.
###############################################
class LangBridgeHarnessModel(Model):
    def __init__(self, model_wrapper, tokenizer):
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer

    def generate(self, context, max_length=128, **kwargs):
        return self.model_wrapper.generate(context, max_length=max_length, **kwargs)
    
    def loglikelihood(self, context, continuation):
        raise NotImplementedError("loglikelihood not implemented for LangBridgeHarnessModel.")


###############################################
# Main evaluation script.
###############################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help="Path to JSON config file")
    args = parser.parse_args()

    # Read configuration
    with open(args.config, "r") as f:
        config = json.load(f)

    # Initialize Weights & Biases logging
    wandb.init(project=config.get("wandb_project", "LangBridge_EvalHarness"), config=config)

    # Set up accelerator for multi-GPU processing
    accelerator = Accelerator()
    device = accelerator.device

    ###############################################
    # 1. Load the encoder model (MT5) and apply PEFT.
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
    # 3. Instantiate the LangBridgeModular model.
    # The alignment and aggregator types are passed via the config.
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

    # Load model checkpoint
    checkpoint_path = config.get("checkpoint_path", "checkpoints/best_model.pt")
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ###############################################
    # 4. Wrap the model for use with the evaluation harness.
    ###############################################
    model_wrapper = LangBridgeModelWrapper(model, decoder_model, tokenizer, device)
    harness_model = LangBridgeHarnessModel(model_wrapper, tokenizer)

    ###############################################
    # 5. Create evaluation tasks based on the decoder model.
    # Depending on the decoder name, select appropriate datasets.
    ###############################################
    tasks = []
    system_prompt = config.get("system_prompt", None)
    decoder_name = config["decoder_model_name_or_path"].lower()
    
    if "metamath" in decoder_name or "llemma" in decoder_name:
        # Math reasoning tasks.
        mgsm_dataset = load_dataset("juletxara/mgsm", split="validation")
        msvamp_dataset = load_dataset("Mathoctopus/MSVAMP", split="validation")
        tasks.append(HFTask(mgsm_dataset, "MGSM", system_prompt=system_prompt))
        tasks.append(HFTask(msvamp_dataset, "MSVAMP", system_prompt=system_prompt))
    elif "codellama" in decoder_name:
        # Code completion tasks.
        humaneval_dataset = load_dataset("openai/openai_humaneval", split="validation")
        tasks.append(HFTask(humaneval_dataset, "HumanEval", system_prompt=system_prompt))
        # Load additional HumanEval-MT datasets from local files.
        mt_files = glob.glob("../humaneval/humaneval_*.json")
        mt_anon_files = glob.glob("../humaneval/humaneval_*_anon.json")
        for f in mt_files:
            lang = os.path.basename(f).split("_")[1].split(".")[0]
            dataset = load_dataset("json", data_files=f, split="train")
            tasks.append(HFTask(dataset, f"HumanEval-MT_{lang}", system_prompt=system_prompt))
        for f in mt_anon_files:
            lang = os.path.basename(f).split("_")[1].split("_")[0]
            dataset = load_dataset("json", data_files=f, split="train")
            tasks.append(HFTask(dataset, f"HumanEval-MT_{lang}_anon", system_prompt=system_prompt))
    elif "orca2" in decoder_name:
        # Logical/commonsense reasoning tasks.
        xcopa_dataset = load_dataset("cambridgeltl/xcopa", split="validation")
        bbh_dataset = load_dataset("lukaemon/bbh", split="validation")
        bbh_bn_dataset = load_dataset(config.get("bbh_bn_eval_dataset", "lukaemon/bbh_bn"), split="validation")
        tasks.append(HFTask(xcopa_dataset, "XCOPA", system_prompt=system_prompt))
        tasks.append(HFTask(bbh_dataset, "BBH", system_prompt=system_prompt))
        tasks.append(HFTask(bbh_bn_dataset, "BBH-BN", system_prompt=system_prompt))
    else:
        # Default task.
        mgsm_dataset = load_dataset("juletxara/mgsm", split="validation")
        tasks.append(HFTask(mgsm_dataset, "MGSM", system_prompt=system_prompt))

    # Log the evaluation tasks to WandB.
    wandb.config.update({"evaluation_tasks": [t.name for t in tasks]})

    ###############################################
    # 6. Run evaluation using the LM Evaluation Harness.
    # Using 0 few-shot examples.
    ###############################################
    print("Running evaluation using LM Evaluation Harness...")
    results = evaluate(harness_model, tasks=tasks, num_fewshot=0, limit=None)

    # Postprocess results for each task.
    final_results = {}
    for task in tasks:
        final_results[task.name] = results.get(task.name, None)

    # Print the evaluation metrics.
    print("Evaluation Results:")
    for task_name, metrics in final_results.items():
        print(f"Task {task_name}:")
        if metrics is not None:
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
        else:
            print("  No results returned.")

    ###############################################
    # 7. Save the evaluation results to a JSON file.
    ###############################################
    eval_output_path = config.get("final_eval_output_path", "final_eval_results.json")
    with open(eval_output_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"Final evaluation results saved to {eval_output_path}")


if __name__ == "__main__":
    main()