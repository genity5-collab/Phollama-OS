# eval/evaluate.py
# Score your Nova models before and after training.
# Run this to see if training actually improved things.
# Never deploy without running this first.

import os
import sys
import json
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import NovaGPT
from src.tokenizer import NovaTokenizer
from src.config import NovaChatConfig, NovaCodeConfig


def run_eval(model_type="chat"):
    """Score the model on the test set. Returns score out of 100."""

    # Load model
    config = NovaChatConfig if model_type == "chat" else NovaCodeConfig
    best   = os.path.join(config.out_dir, "best.pt")
    final  = os.path.join(config.out_dir, "final.pt")
    path   = best if os.path.exists(best) else final

    if not os.path.exists(path):
        print(f"No model found at {path}. Train first.")
        return 0

    model, ckpt = NovaGPT.load_checkpoint(path, config)
    model.eval()

    tokenizer = NovaTokenizer("models/nova_tokenizer.json")
    system = (
        "You are Nova, an honest and helpful AI assistant."
        if model_type == "chat"
        else "You are Nova Code, a coding assistant."
    )

    # Load test cases
    test_file = f"eval/test_{model_type}.json"
    with open(test_file) as f:
        tests = json.load(f)

    print(f"\nEvaluating Nova {model_type} ({len(tests)} test cases)")
    print(f"Model: {path}")
    print(f"Step: {ckpt.get('step', '?')} | Loss: {ckpt.get('loss', '?'):.4f}")
    print("-" * 60)

    passed = 0
    failed = 0
    results_by_category = {}

    for test in tests:
        category = test["category"]
        prompt_text = test["prompt"]
        must_contain = test.get("must_contain", [])

        # Build prompt
        prompt = (
            f"<|system|>{system}\n"
            f"<|user|>{prompt_text}\n"
            f"<|assistant|>"
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        idx = torch.tensor([ids])

        # Generate
        with torch.no_grad():
            out = model.generate(
                idx,
                max_new_tokens=100,
                temperature=0.7,
                top_k=40,
            )

        new_ids  = out[0][len(ids):].tolist()
        response = tokenizer.decode(new_ids).lower()

        # Stop at end markers
        for stop in ["<|user|>", "<|system|>", "<eos>"]:
            if stop in response:
                response = response[:response.index(stop)]
        response = response.strip()

        # Check must_contain
        if not must_contain:
            ok = len(response) > 5  # just needs a response
        else:
            ok = any(kw.lower() in response for kw in must_contain)

        if ok:
            passed += 1
            status = "✅"
        else:
            failed += 1
            status = "❌"

        if category not in results_by_category:
            results_by_category[category] = {"passed": 0, "total": 0}
        results_by_category[category]["total"] += 1
        if ok:
            results_by_category[category]["passed"] += 1

        # Print failures in detail
        if not ok:
            print(f"{status} [{category}] {prompt_text}")
            print(f"   Expected one of: {must_contain}")
            print(f"   Got: {response[:100]}")

    total = passed + failed
    score = round(passed / total * 100)

    print("\n" + "=" * 60)
    print(f"SCORE: {passed}/{total} = {score}/100")
    print("\nBy category:")
    for cat, r in sorted(results_by_category.items()):
        cat_score = round(r['passed'] / r['total'] * 100)
        bar = "█" * (cat_score // 10) + "░" * (10 - cat_score // 10)
        print(f"  {cat:12s} {bar} {cat_score}%")

    if score >= 75:
        print(f"\n✅ PASS ({score}/100) - model is good to deploy")
    elif score >= 50:
        print(f"\n⚠️  BORDERLINE ({score}/100) - needs more training data")
    else:
        print(f"\n❌ FAIL ({score}/100) - do not deploy, keep training")

    return score


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chat", choices=["chat", "code"])
    args = parser.parse_args()
    run_eval(args.model)
