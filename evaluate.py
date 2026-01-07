"""
Evaluation Script for GRPO-trained model
Compares baseline Qwen vs fine-tuned model on GSM8K
"""

import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime


# =============================================================================
# Helper Functions (same as training)
# =============================================================================

SYSTEM_PROMPT = """You are a helpful math tutor. Solve problems step by step.
Always show your reasoning, then give the final numerical answer."""


def extract_answer_from_gsm8k(answer_text):
    """Extract the final numerical answer from GSM8K format."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if match:
        return float(match.group(1).replace(",", ""))
    return None


def extract_answer_from_model(model_output):
    """Extract the last number from model's response."""
    numbers = re.findall(r"-?[\d,]+\.?\d*", model_output)
    numbers = [n for n in numbers if n and n.strip(",-.")]
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            return None
    return None


def count_reasoning_steps(text):
    """Count reasoning steps in response."""
    step_patterns = [
        r"[Ss]tep\s*\d+",
        r"^\d+\.",
        r"[Ff]irst[,:]",
        r"[Ss]econd[,:]",
        r"[Tt]hird[,:]",
        r"[Tt]hen[,:]",
        r"[Nn]ext[,:]",
        r"[Ff]inally[,:]",
    ]
    count = 0
    for pattern in step_patterns:
        count += len(re.findall(pattern, text, re.MULTILINE))
    return count


def format_prompt(question, tokenizer):
    """Format question for the model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(model, tokenizer, test_data, model_name="Model"):
    """Evaluate a model on test data."""

    results = {
        "correct": 0,
        "total": 0,
        "has_steps": 0,
        "total_steps": 0,
        "responses": []
    }

    print(f"\nEvaluating {model_name}...")

    for i, example in enumerate(test_data):
        question = example["question"]
        ground_truth = example["answer"]
        expected = extract_answer_from_gsm8k(ground_truth)

        # Generate response
        prompt = format_prompt(question, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # Greedy for consistency
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode response (only the generated part)
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # Extract answer and count steps
        predicted = extract_answer_from_model(response)
        num_steps = count_reasoning_steps(response)

        # Check correctness
        is_correct = False
        if predicted is not None and expected is not None:
            is_correct = abs(predicted - expected) < 1e-5

        # Update stats
        results["total"] += 1
        if is_correct:
            results["correct"] += 1
        if num_steps >= 1:
            results["has_steps"] += 1
        results["total_steps"] += num_steps

        # Store response details
        results["responses"].append({
            "question": question[:100] + "...",
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "num_steps": num_steps,
            "response": response[:200] + "..." if len(response) > 200 else response
        })

        status = "OK" if is_correct else "X"
        print(
            f"  [{i+1}/{len(test_data)}] {status} Expected: {expected}, Got: {predicted}, Steps: {num_steps}")

    # Calculate metrics
    results["accuracy"] = results["correct"] / \
        results["total"] if results["total"] > 0 else 0
    results["step_rate"] = results["has_steps"] / \
        results["total"] if results["total"] > 0 else 0
    results["avg_steps"] = results["total_steps"] / \
        results["total"] if results["total"] > 0 else 0

    return results


def save_results(all_results, filename="evaluation_results.txt"):
    """Save evaluation results to file."""

    with open(filename, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Model Comparison: Baseline vs GRPO vs DPO\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        # Summary table
        f.write("SUMMARY\n")
        f.write("-" * 70 + "\n")
        header = f"{'Metric':<25}"
        for name in all_results:
            header += f" {name:<14}"
        f.write(header + "\n")
        f.write("-" * 70 + "\n")

        # Accuracy row
        row = f"{'Accuracy':<25}"
        for name, res in all_results.items():
            row += f" {res['accuracy']*100:>6.1f}%{'':<7}"
        f.write(row + "\n")

        # Steps row
        row = f"{'Avg Steps':<25}"
        for name, res in all_results.items():
            row += f" {res['avg_steps']:>6.2f}{'':<8}"
        f.write(row + "\n")

        # Correct row
        row = f"{'Correct / Total':<25}"
        for name, res in all_results.items():
            row += f" {res['correct']}/{res['total']:<12}"
        f.write(row + "\n")

        f.write("-" * 70 + "\n\n")

        # Detailed responses
        f.write("=" * 70 + "\n")
        f.write("DETAILED RESPONSES\n")
        f.write("=" * 70 + "\n\n")

        for model_name, results in all_results.items():
            f.write(f"\n--- {model_name} ---\n\n")
            for i, resp in enumerate(results["responses"]):
                f.write(f"Example {i+1}:\n")
                f.write(f"  Question: {resp['question']}\n")
                f.write(f"  Expected: {resp['expected']}\n")
                f.write(f"  Predicted: {resp['predicted']}\n")
                f.write(f"  Correct: {resp['correct']}\n")
                f.write(f"  Steps: {resp['num_steps']}\n")
                f.write(f"  Response: {resp['response']}\n\n")

    print(f"\nResults saved to {filename}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import os

    # Load test dataset
    print("Loading GSM8K test set...")
    dataset = load_dataset("openai/gsm8k", "main")
    test_data = dataset["test"].select(range(30))  # 30 questions
    print(f"Using {len(test_data)} test examples")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    # Evaluate baseline model
    print("\n" + "=" * 50)
    print("Loading Baseline Model (Qwen 0.5B)")
    print("=" * 50)
    baseline_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    all_results["Baseline"] = evaluate_model(
        baseline_model, tokenizer, test_data, "Baseline")
    del baseline_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Evaluate GRPO trained model
    if os.path.exists("./grpo_output"):
        print("\n" + "=" * 50)
        print("Loading GRPO Trained Model")
        print("=" * 50)
        grpo_model = AutoModelForCausalLM.from_pretrained(
            "./grpo_output",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        all_results["GRPO"] = evaluate_model(
            grpo_model, tokenizer, test_data, "GRPO Trained")
        del grpo_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Evaluate DPO trained model
    if os.path.exists("./dpo_output"):
        print("\n" + "=" * 50)
        print("Loading DPO Trained Model")
        print("=" * 50)
        dpo_model = AutoModelForCausalLM.from_pretrained(
            "./dpo_output",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        all_results["DPO"] = evaluate_model(
            dpo_model, tokenizer, test_data, "DPO Trained")
        del dpo_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Print summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"\n{'Model':<20} {'Accuracy':<12} {'Avg Steps':<12}")
    print("-" * 44)
    for name, res in all_results.items():
        print(
            f"{name:<20} {res['accuracy']*100:.1f}%{'':<7} {res['avg_steps']:.2f}")

    # Save results
    save_results(all_results)

