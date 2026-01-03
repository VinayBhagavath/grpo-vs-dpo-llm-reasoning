"""
DPO Training Script for Step-by-Step Reasoning
Fine-tunes Qwen 0.5B on GSM8K math problems using DPO
"""

import re
import sys
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


# =============================================================================
# Step 2: Load Dataset
# =============================================================================

def load_gsm8k():
    """Load GSM8K dataset for math reasoning training."""
    dataset = load_dataset("openai/gsm8k", "main")

    print(f"Train examples: {len(dataset['train'])}")
    print(f"Test examples: {len(dataset['test'])}")

    # Show a sample
    sample = dataset["train"][0]
    print("\n--- Sample Problem ---")
    print(f"Question: {sample['question'][:200]}...")
    print(f"Answer: {sample['answer'][:200]}...")

    return dataset


# =============================================================================
# Step 3: Load Model and Tokenizer
# =============================================================================

def load_model_and_tokenizer(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """Load Qwen 0.5B model and tokenizer."""

    print(f"\nLoading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    print(f"Model loaded on: {next(model.parameters()).device}")
    print(f"Model parameters: {model.num_parameters():,}")

    return model, tokenizer


# =============================================================================
# Step 4: Create Preference Data
# =============================================================================

def extract_answer_from_gsm8k(answer_text):
    """Extract the final numerical answer from GSM8K format."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    return None


def clean_gsm8k_answer(answer_text):
    """Clean GSM8K answer to be a proper step-by-step response."""
    # Remove the calculator annotations like <<48/2=24>>
    cleaned = re.sub(r"<<[^>]+>>", "", answer_text)
    # Remove the #### final answer marker
    cleaned = re.sub(r"####\s*\d+", "", cleaned)
    return cleaned.strip()


def create_preference_dataset(dataset, tokenizer):
    """
    Convert GSM8K to DPO preference format.

    Chosen: Full step-by-step reasoning (from GSM8K)
    Rejected: Just the final answer (no reasoning)
    """
    preference_data = []

    for example in dataset:
        question = example["question"]
        full_answer = example["answer"]

        # Extract components
        final_number = extract_answer_from_gsm8k(full_answer)
        step_by_step = clean_gsm8k_answer(full_answer)

        if final_number is None:
            continue

        # Format the prompt
        prompt = format_prompt(question, tokenizer)

        # Chosen: Step-by-step reasoning with final answer
        chosen = f"{step_by_step}\n\nFinal Answer: {final_number}"

        # Rejected: Just the number (no reasoning)
        rejected = f"The answer is {final_number}."

        preference_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    return Dataset.from_list(preference_data)


# =============================================================================
# Step 5: Prompt Formatting
# =============================================================================

SYSTEM_PROMPT = """You are a helpful math tutor. Solve problems step by step.
Always show your reasoning, then give the final numerical answer."""


def format_prompt(question, tokenizer):
    """Format a GSM8K question into a chat prompt for the model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    # Use the tokenizer's chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt


# =============================================================================
# Step 6: DPO Training Configuration
# =============================================================================

def get_training_config(output_dir="./dpo_output"):
    """Create DPO training configuration."""

    # Check if CUDA is available
    use_bf16 = torch.cuda.is_available()

    config = DPOConfig(
        output_dir=output_dir,

        # Training settings
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=5e-6,

        # DPO specific
        beta=0.1,  # KL penalty coefficient
        max_length=512,
        max_prompt_length=256,

        # Precision
        bf16=use_bf16,

        # Logging
        logging_steps=10,
        save_steps=500,

        # Disable wandb
        report_to="none",
    )

    return config


# =============================================================================
# Step 7: Training
# =============================================================================

def train(model, tokenizer, train_dataset, config):
    """Run DPO training."""

    # Initialize trainer
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("Starting DPO training...")
    trainer.train()

    # Save model
    trainer.save_model(config.output_dir)
    print(f"Model saved to {config.output_dir}")

    return trainer


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Load dataset
    print("=" * 50)
    print("Loading GSM8K Dataset")
    print("=" * 50)
    dataset = load_gsm8k()

    # Test preference data creation
    print("\n" + "=" * 50)
    print("Testing Preference Data Creation")
    print("=" * 50)

    # Load tokenizer for formatting
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create a small preference dataset for demo
    small_dataset = dataset["train"].select(range(3))
    pref_data = create_preference_dataset(small_dataset, tokenizer)

    print(f"\nCreated {len(pref_data)} preference pairs")
    print("\n--- Sample Preference Pair ---")
    print(f"Prompt (truncated): {pref_data[0]['prompt'][:100]}...")
    print(f"\nChosen: {pref_data[0]['chosen'][:150]}...")
    print(f"\nRejected: {pref_data[0]['rejected']}")

    # Test DPO config
    print("\n" + "=" * 50)
    print("DPO Training Config")
    print("=" * 50)

    config = get_training_config()
    print(f"  Output dir: {config.output_dir}")
    print(f"  Beta (KL coef): {config.beta}")
    print(f"  Max length: {config.max_length}")
    print(f"  Batch size: {config.per_device_train_batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_train_epochs}")

    # Check if --train flag is passed
    if "--train" in sys.argv:
        print("\n" + "=" * 50)
        print("Starting Training")
        print("=" * 50)

        # Load model
        model, tokenizer = load_model_and_tokenizer()

        # Use small subset if --small flag
        train_data = dataset["train"]
        if "--small" in sys.argv:
            train_data = train_data.select(range(30))
            print(f"Using small dataset: {len(train_data)} examples")

        # Create preference dataset
        train_dataset = create_preference_dataset(train_data, tokenizer)
        print(f"Created {len(train_dataset)} preference pairs")

        # Train
        train(model, tokenizer, train_dataset, config)
    else:
        print("\n" + "=" * 50)
        print("Setup Complete! Run with --train to start training")
        print("=" * 50)
