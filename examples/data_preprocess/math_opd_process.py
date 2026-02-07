import os
import datasets

# =========================
# 路径配置
# =========================
BASE_DIR = "/data/home/zdhs0086/hhh/verl-agent/data/math"

TRAIN_IN = os.path.join(BASE_DIR, "dapo-math-17k.parquet")
TEST_IN = os.path.join(BASE_DIR, "aime-2024.parquet")

TRAIN_OUT = os.path.join(BASE_DIR, "train.parquet")
TEST_OUT = os.path.join(BASE_DIR, "test.parquet")



PREFIX = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form Answer: "
)

SUFFIX = 'Remember to put your answer on its own line after "Answer:".'


def clean_prompt(prompt: str) -> str:
    """
    按“行”清洗 prompt：
    - 删除 instruction 行
    - 保留真正的题目正文
    """
    lines = [line.rstrip() for line in prompt.strip().splitlines()]

    cleaned_lines = []
    for line in lines:
        stripped = line.strip()

        # 跳过明显的 instruction 行
        if not stripped:
            continue

        if stripped.startswith("Solve the following math problem"):
            continue

        if stripped.startswith("The last line of your response should be of the form"):
            continue

        if stripped.startswith("Remember to put your answer on its own line"):
            continue


        if stripped.startswith("$Answer (without quotes) where $Answer"):
            continue

        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines).strip()

    return result



# =========================
# 通用 map 函数
# =========================
def make_map_fn(split: str, data_source: str):
    def process_fn(example, idx):
        # -------- prompt --------
        raw_prompt = example["prompt"][0]["content"]
        question = clean_prompt(raw_prompt)


        if question.startswith("$Answer"):
            raise ValueError(
            f"[BAD PROMPT] Instruction residue detected at idx={idx}"
        )
        # 强校验：绝不允许空题
        if not question.strip():
            raise ValueError(
                f"[FATAL] Empty question after cleaning! "
                f"split={split}, idx={idx}"
            )

        # -------- ground truth --------
        reward_model = example["reward_model"]
        ground_truth = str(reward_model["ground_truth"])

        return {
            "data_source": data_source,
            "ability": "math",
            "reward_model": {
                "style": reward_model.get("style", "rule"),
                "ground_truth": ground_truth,
            },
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "extra_info": {
                "split": split,
                "index": idx,
                "question": question,
                "answer": ground_truth,
            },
            "env_kwargs": {
                "question": question,
                "ground_truth": ground_truth,
                "data_source": data_source,
            },
        }


    return process_fn


# =========================
# 主流程
# =========================
if __name__ == "__main__":
    # ---------- Train ----------
    train_ds = datasets.load_dataset(
        "parquet",
        data_files=TRAIN_IN,
        split="train",
    )

    train_ds = train_ds.map(
        make_map_fn("train", "dapo-math-17k"),
        with_indices=True,
        remove_columns=train_ds.column_names,
    )

    train_ds.to_parquet(TRAIN_OUT)
    print(f"[OK] Train saved to {TRAIN_OUT} | size={len(train_ds)}")

    # ---------- Test ----------
    test_ds = datasets.load_dataset(
        "parquet",
        data_files=TEST_IN,
        split="train",
    )

    test_ds = test_ds.map(
        make_map_fn("test", "aime-2024"),
        with_indices=True,
        remove_columns=test_ds.column_names,
    )

    test_ds.to_parquet(TEST_OUT)
    print(f"[OK] Test saved to {TEST_OUT} | size={len(test_ds)}")
