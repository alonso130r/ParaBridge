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

# Import our LangBridge model definition (assumed defined in model.py)
from model import LangBridgeWithLSTM


###############################################
# Custom Task class for Hugging Face datasets.
# Assumes each example is a dict with keys "prompt" and "answer".
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
        else:
            return doc["prompt"]

    def doc_to_target(self, doc):
        # Expected answer remains as stored.
        return doc["answer"].strip()

    def postprocess(self, generated_text):
        # Remove the system prompt from the generated text if present.
        if self.system_prompt and generated_text.startswith(self.system_prompt):
            return generated_text[len(self.system_prompt):].strip()
        return generated_text.strip()

    def aggregation(self):
        # Return a metric function that computes accuracy.
        return {"accuracy": lambda items: sum(items) / len(items) if items else 0.0}

    def higher_is_better(self):
        return {"accuracy": True}

    def fewshot_description(self):
        return "0-shot evaluation"


###############################################
# Helper wrapper for our LangBridge model.
###############################################
class LangBridgeModelWrapper:
    def __init__(self, model, decoder_model, tokenizer, device):
        self.model = model
        self.decoder_model = decoder_model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, prompt, max_length=128, **kwargs):
        self.model.eval()
        with torch.no_grad():
            # Our LangBridge model expects a list of prompts.
            _, soft_prompt = self.model([prompt])
            generated_ids = self.decoder_model.generate(
                inputs_embeds=soft_prompt,
                max_length=max_length,
                **kwargs
            )
            generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()


###############################################
# A wrapper to expose our model in the format expected
# by the LM Evaluation Harness.
###############################################
class LangBridgeHarnessModel(Model):
    def __init__(self, model_wrapper, tokenizer):
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer

    def generate(self, context, max_length=128, **kwargs):
        # Generate text for a given prompt (context).
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

    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)

    # Initialize WandB logging
    wandb.init(project=config.get("wandb_project", "LangBridge_EvalHarness"), config=config)

    # Setup accelerator for multi-GPU processing
    accelerator = Accelerator()
    device = accelerator.device

    ###############################################
    # 1. Load the encoder (MT5) from a fixed path and wrap with PEFT.
    ###############################################
    encoder_variant = config.get("encoder_variant", "AMS")  # "AMS" or "barlow"
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
    # 2. Load the decoder model from the specified path.
    ###############################################
    decoder_model = AutoModel.from_pretrained(config["decoder_model_name_or_path"]).to(device)

    ###############################################
    # 3. Instantiate the LangBridge model.
    ###############################################
    model = LangBridgeWithLSTM(
        encoder_model=encoder_model,
        decoder_model=decoder_model,
        tokenizer=tokenizer,
        fine_tune_encoder=config.get("fine_tune_encoder", True),
        lstm_num_layers=config.get("lstm_num_layers", 1),
        max_sentence_length=config.get("max_sentence_length", 32),
        prompt_length=config.get("prompt_length", 10)
    ).to(device)

    # Load the saved checkpoint.
    checkpoint_path = config.get("checkpoint_path", "checkpoints/best_model.pt")
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ###############################################
    # 4. Wrap the model for generation.
    ###############################################
    model_wrapper = LangBridgeModelWrapper(model, decoder_model, tokenizer, device)
    harness_model = LangBridgeHarnessModel(model_wrapper, tokenizer)

    ###############################################
    # 5. Create evaluation tasks based on decoder model.
    # Use the following datasets (loaded directly from HF or locally):
    #   - XCOPA: https://huggingface.co/datasets/cambridgeltl/xcopa
    #   - MSVAMP: https://huggingface.co/datasets/Mathoctopus/MSVAMP
    #   - BBH: https://huggingface.co/datasets/lukaemon/bbh
    #   - BBH-BN: (default id: "lukaemon/bbh_bn")
    #   - HumanEval: https://huggingface.co/datasets/openai/openai_humaneval
    #   - HumanEval-MT: from local directory /humaneval (files separated by language)
    #   - MGSM: https://huggingface.co/datasets/juletxara/mgsm
    ###############################################
    tasks = []
    system_prompt = config.get("system_prompt", None)
    decoder_name = config["decoder_model_name_or_path"].lower()
    
    if "metamath" in decoder_name or "llemma" in decoder_name:
        # For math reasoning tasks: MGSM and MSVAMP.
        mgsm_dataset = load_dataset("juletxara/mgsm", split="validation")
        msvamp_dataset = load_dataset("Mathoctopus/MSVAMP", split="validation")
        tasks.append(HFTask(mgsm_dataset, "MGSM", system_prompt=system_prompt))
        tasks.append(HFTask(msvamp_dataset, "MSVAMP", system_prompt=system_prompt))
    elif "codellama" in decoder_name:
        # For code completion tasks: HumanEval and HumanEval-MT.
        humaneval_dataset = load_dataset("openai/openai_humaneval", split="validation")
        tasks.append(HFTask(humaneval_dataset, "HumanEval", system_prompt=system_prompt))
        # Load HumanEval-MT files from local directory /humaneval.
        import glob
        mt_files = glob.glob("/humaneval/humaneval_*.json")
        mt_anon_files = glob.glob("/humaneval/humaneval_*_anon.json")
        for f in mt_files:
            basename = os.path.basename(f)
            # Assume format: humaneval_{lang}.json
            lang = basename.split("_")[1].split(".")[0]
            dataset = load_dataset("json", data_files=f, split="train")
            tasks.append(HFTask(dataset, f"HumanEval-MT_{lang}", system_prompt=system_prompt))
        for f in mt_anon_files:
            basename = os.path.basename(f)
            # Assume format: humaneval_{lang}_anon.json
            lang = basename.split("_")[1].split("_")[0]
            dataset = load_dataset("json", data_files=f, split="train")
            tasks.append(HFTask(dataset, f"HumanEval-MT_{lang}_anon", system_prompt=system_prompt))
    elif "orca2" in decoder_name:
        # For logical/commonsense reasoning: XCOPA, BBH, and BBH-BN.
        xcopa_dataset = load_dataset("cambridgeltl/xcopa", split="validation")
        bbh_dataset = load_dataset("lukaemon/bbh", split="validation")
        bbh_bn_dataset = load_dataset(config.get("bbh_bn_eval_dataset", "lukaemon/bbh_bn"), split="validation")
        tasks.append(HFTask(xcopa_dataset, "XCOPA", system_prompt=system_prompt))
        tasks.append(HFTask(bbh_dataset, "BBH", system_prompt=system_prompt))
        tasks.append(HFTask(bbh_bn_dataset, "BBH-BN", system_prompt=system_prompt))
    else:
        # Default to using MGSM.
        mgsm_dataset = load_dataset("juletxara/mgsm", split="validation")
        tasks.append(HFTask(mgsm_dataset, "MGSM", system_prompt=system_prompt))

    # Log task names to wandb.
    wandb.config.update({"evaluation_tasks": [t.name for t in tasks]})

    ###############################################
    # 6. Run evaluation using the EleutherAI LM Evaluation Harness.
    # We use 0 few-shot examples.
    ###############################################
    print("Running evaluation using LM Evaluation Harness...")
    results = evaluate(harness_model, tasks=tasks, num_fewshot=0, limit=None)

    # Postprocess generated outputs for each task using each task's postprocess() method.
    # (The evaluation harness may already support this if the task defines a postprocess method.)
    final_results = {}
    for task in tasks:
        # Assume results contains a dict with key equal to task.name.
        if task.name in results:
            # Optionally, you can apply postprocessing on the generated outputs here.
            final_results[task.name] = results[task.name]
        else:
            final_results[task.name] = None

    print("Evaluation Results:")
    for task_name, metrics in final_results.items():
        print(f"Task {task_name}:")
        if metrics is not None:
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
        else:
            print("  No results returned.")

    ###############################################
    # 7. Save the evaluation results.
    ###############################################
    eval_output_path = config.get("final_eval_output_path", "final_eval_results.json")
    with open(eval_output_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"Final evaluation results saved to {eval_output_path}")

if __name__ == "__main__":
    main()
