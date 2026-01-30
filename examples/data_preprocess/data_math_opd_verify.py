import datasets
import random

BASE_DIR = "/data/home/zdhs0086/hhh/verl-agent/data/math"

TRAIN_FILE = f"{BASE_DIR}/train.parquet"
TEST_FILE = f"{BASE_DIR}/test.parquet"


def verify_one(name, path, num_samples=5):
    print("=" * 80)
    print(f"[Verify] {name}")
    print("=" * 80)

    ds = datasets.load_dataset(
        "parquet",
        data_files=path,
        split="train",
    )

    print("Dataset size:", len(ds))
    print("Columns:", ds.column_names)
    print()

    indices = random.sample(range(len(ds)), num_samples)

    for idx in indices:
        ex = ds[idx]
        question = ex["prompt"][0]["content"]
        answer = ex["reward_model"]["ground_truth"]

        print(f"--- Sample idx={idx} ---")
        print("Question:")
        print(question[:500])
        print()
        print("Ground truth:", answer)
        print()

        # 强校验
        assert question.strip(), "❌ Empty question detected!"
        assert answer is not None, "❌ Missing ground truth!"

    print("[PASS] All sampled examples look good.\n")


FILES = {
    "train": f"{BASE_DIR}/train.parquet",
    "test": f"{BASE_DIR}/test.parquet",
}

COT_KEYWORDS = [
    "Please carefully reason through the math problem step by step",
    "reason through the math problem step by step",
    "final answer within \\boxed",
]


def verify_file(name, path):
    print("=" * 80)
    print(f"[VERIFY] {name}")
    print("=" * 80)

    ds = datasets.load_dataset(
        "parquet",
        data_files=path,
        split="train",
    )

    hit = 0
    for i, ex in enumerate(ds):
        prompt = ex["prompt"][0]["content"]

        for kw in COT_KEYWORDS:
            if kw in prompt:
                print(f"[FOUND] idx={i}")
                print(prompt[:500])
                print()
                hit += 1
                break

    if hit == 0:
        print("[PASS] No CoT instruction found in dataset ✅")
    else:
        print(f"[FAIL] Found {hit} examples with CoT instruction ❌")


# if __name__ == "__main__":
#     for name, path in FILES.items():
#         verify_file(name, path)


if __name__ == "__main__":
    verify_one("TRAIN", TRAIN_FILE)
    verify_one("TEST", TEST_FILE)
