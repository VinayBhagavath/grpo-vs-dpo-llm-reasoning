"""
GRPO Training Script for Step-by-Step Reasoning
Fine-tunes Qwen 0.5B on GSM8K math problems using GRPO
"""

import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


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
# Step 4: Reward Function
# =============================================================================

def extract_answer_from_gsm8k(answer_text):
    """Extract the final numerical answer from GSM8K format.

    GSM8K answers end with '#### <number>'
    """
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if match:
        # Remove commas and convert to float
        return float(match.group(1).replace(",", ""))
    return None


def extract_answer_from_model(model_output):
    """Extract the last number from model's response as the answer."""
    # Find all numbers in the output (including negatives and decimals)
    # Must have at least one digit
    numbers = re.findall(r"-?[\d,]+\.?\d*", model_output)
    # Filter out empty matches and standalone punctuation
    numbers = [n for n in numbers if n and n.strip(",-. ")]
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            return None
    return None


def count_reasoning_steps(text):
    """Count how many reasoning steps are in the response."""
    # Look for patterns like "Step 1", "Step 2", "First,", "Second,", "1.", "2." etc.
    step_patterns = [
        r"[Ss]tep\s*\d+",        # "Step 1", "step 2"
        r"^\d+\.",               # "1.", "2." at start of line
        r"[Ff]irst[,:]",         # "First,"
        r"[Ss]econd[,:]",        # "Second,"
        r"[Tt]hird[,:]",         # "Third,"
        r"[Tt]hen[,:]",          # "Then,"
        r"[Nn]ext[,:]",          # "Next,"
        r"[Ff]inally[,:]",       # "Finally,"
    ]

    count = 0
    for pattern in step_patterns:
        count += len(re.findall(pattern, text, re.MULTILINE))

    return count


def reward_function(prompts, completions, ground_truths):
    """
    Compute rewards for model completions.

    Rewards both correctness AND step-by-step reasoning.

    Args:
        prompts: List of input prompts (unused but required by GRPO)
        completions: List of model-generated completions
        ground_truths: List of correct answers (from GSM8K)

    Returns:
        List of rewards
    """
    rewards = []

    for completion, truth in zip(completions, ground_truths):
        reward = 0.0

        # === Reward 1: Step-by-step reasoning (up to +1.0) ===
        num_steps = count_reasoning_steps(completion)
        if num_steps >= 2:
            reward += 1.0   # Good: multiple steps
        elif num_steps == 1:
            reward += 0.5   # Partial: at least one step
        # else: no steps, no bonus

        # === Reward 2: Correct answer (+2.0) ===
        predicted = extract_answer_from_model(completion)
        expected = extract_answer_from_gsm8k(truth)

        if predicted is not None and expected is not None:
            if abs(predicted - expected) < 1e-5:
                reward += 2.0  # Correct answer is worth more

        # Normalize: scale to roughly [-1, 1] range
        # Max reward = 3.0 (steps + correct), min = 0.0
        # Shift to center around 0: subtract 1.5, then scale
        reward = (reward - 1.5) / 1.5  # Maps [0, 3] -> [-1, 1]

        rewards.append(reward)

    return rewards


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


def prepare_dataset(dataset, tokenizer):
    """Prepare the dataset with formatted prompts for GRPO training."""

    def format_example(example):
        return {
            "prompt": format_prompt(example["question"], tokenizer),
            "answer": example["answer"],  # Keep for reward computation
        }

    # Apply formatting to all examples
    formatted = dataset.map(
        format_example, remove_columns=dataset.column_names)
    return formatted


# =============================================================================
# Step 6: GRPO Training Configuration
# =============================================================================

def get_training_config(output_dir="./grpo_output"):
    """Create GRPO training configuration."""

    # Check if CUDA is available
    use_bf16 = torch.cuda.is_available()

    config = GRPOConfig(
        output_dir=output_dir,

        # Generation settings
        max_completion_length=256,     # Max tokens for model response
        num_generations=4,             # Completions per prompt for comparison

        # Training settings
        per_device_train_batch_size=1,  # Small batch for memory
        gradient_accumulation_steps=4,  # Effective batch = 4
        num_train_epochs=1,            # Quick training run
        learning_rate=5e-6,            # Small LR for stability

        # Precision - use bf16 only if GPU available
        bf16=use_bf16,

        # Logging
        logging_steps=10,
        save_steps=500,

        # Disable wandb by default
        report_to="none",
    )

    return config


# =============================================================================
# Step 7: Training
# =============================================================================

def create_reward_fn(tokenizer, dataset):
    """Create reward function for GRPO trainer.

    GRPO calls this with: completions, prompts (optional)
    We need to look up the ground truth answer for each prompt.
    """
    # Build lookup from prompt to answer
    prompt_to_answer = {}
    for example in dataset:
        prompt_to_answer[example["prompt"]] = example["answer"]

    def reward_fn(completions, prompts=None, **kwargs):
        # Get ground truths for these prompts
        ground_truths = [prompt_to_answer.get(p, "") for p in prompts]
        return reward_function(prompts, completions, ground_truths)

    return reward_fn


def train(model, tokenizer, train_dataset, config):
    """Run GRPO training."""

    # Create reward function
    reward_fn = create_reward_fn(tokenizer, train_dataset)

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

    # Train
    print("Starting GRPO training...")
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

    # Test reward function
    print("\n" + "=" * 50)
    print("Testing Reward Function")
    print("=" * 50)

    # Test cases - showing how steps + correctness both matter
    sample_answer = dataset["train"][0]["answer"]  # "... #### 72"

    test_completions = [
        # Best: correct answer WITH steps
        "Step 1: Find clips in May: 48/2 = 24. Step 2: Total = 48 + 24 = 72",
        # Good: correct answer, no steps
        "72",
        # Partial: wrong answer, but has steps
        "Step 1: 48/2 = 24. Step 2: 48 - 24 = 24. The answer is 24",
        # Bad: wrong answer, no steps
        "100",
    ]

    print(f"Ground truth: {extract_answer_from_gsm8k(sample_answer)}\n")

    rewards = reward_function(
        prompts=[""] * len(test_completions),
        completions=test_completions,
        ground_truths=[sample_answer] * len(test_completions)
    )

    for completion, reward in zip(test_completions, rewards):
        steps = count_reasoning_steps(completion)
        print(
            f"  Steps: {steps} | Reward: {reward:+.2f} | '{completion[:45]}...'")

    # Test prompt formatting
    print("\n" + "=" * 50)
    print("Testing Prompt Formatting")
    print("=" * 50)

    # Load tokenizer for formatting
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    sample_question = dataset["train"][0]["question"]
    formatted_prompt = format_prompt(sample_question, tokenizer)

    print("Original question:")
    print(f"  {sample_question}\n")
    print("Formatted prompt:")
    print(formatted_prompt)

    # Test GRPO config
    print("\n" + "=" * 50)
    print("GRPO Training Config")
    print("=" * 50)

    config = get_training_config()
    print(f"  Output dir: {config.output_dir}")
    print(f"  Max completion length: {config.max_completion_length}")
    print(f"  Num generations: {config.num_generations}")
    print(f"  Batch size: {config.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_train_epochs}")

    # Check if --train flag is passed
    import sys
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

        # Prepare dataset
        train_dataset = prepare_dataset(train_data, tokenizer)

        # Train
        train(model, tokenizer, train_dataset, config)
    else:
        print("\n" + "=" * 50)
        print("Setup Complete! Run with --train to start training")
        print("=" * 50)
