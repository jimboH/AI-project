import json
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

DATASETS = [
    {
        "jsonl": "/work/u1304848/AI/project/datasets/meta_All_Beauty.jsonl",
        "output_dir": "/work/u1304848/AI/project/datasets/images/All_Beauty",
        "name": "All_Beauty",
    },
    {
        "jsonl": "/work/u1304848/AI/project/datasets/meta_Musical_Instruments.jsonl",
        "output_dir": "/work/u1304848/AI/project/datasets/images/Musical_Instruments",
        "name": "Musical_Instruments",
    },
    {
        "jsonl": "/work/u1304848/AI/project/datasets/meta_Sports_and_Outdoors.jsonl",
        "output_dir": "/work/u1304848/AI/project/datasets/images/Sports_and_Outdoors",
        "name": "Sports_and_Outdoors",
    },
]

FAILED_LOG_DIR = "/work/u1304848/AI/project/datasets"
MAX_WORKERS = 16


def download_image(url, dest_path):
    if os.path.exists(dest_path):
        return dest_path, url, "skipped"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(response.content)
        return dest_path, url, "ok"
    except Exception as e:
        return dest_path, url, f"error: {e}"


def collect_tasks(jsonl_path, output_dir):
    tasks = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            parent_asin = record.get("parent_asin")
            images = record.get("images", [])
            for img in images:
                url = img.get("large")
                variant = img.get("variant")
                if not url or not variant or not parent_asin:
                    continue
                filename = f"{parent_asin}_{variant}.jpg"
                dest_path = os.path.join(output_dir, filename)
                tasks.append((url, dest_path))
    return tasks


def process_dataset(dataset):
    name = dataset["name"]
    jsonl_path = dataset["jsonl"]
    output_dir = dataset["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    failed_log_path = os.path.join(FAILED_LOG_DIR, f"failed_downloads_{name}.jsonl")

    print(f"\n[{name}] Collecting image tasks from {jsonl_path} ...")
    tasks = collect_tasks(jsonl_path, output_dir)
    print(f"[{name}] Total image entries: {len(tasks)}")

    ok = skipped = errors = 0
    failed_records = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_image, url, path): (url, path)
            for url, path in tasks
        }
        with tqdm(total=len(tasks), desc=name, unit="img") as bar:
            for future in as_completed(futures):
                path, url, status = future.result()
                if status == "ok":
                    ok += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    errors += 1
                    failed_records.append({"url": url, "dest_path": path, "error": status})
                    tqdm.write(f"  FAILED {os.path.basename(path)}: {status}")
                bar.set_postfix(ok=ok, skipped=skipped, errors=errors)
                bar.update(1)

    if failed_records:
        with open(failed_log_path, "w") as f:
            for record in failed_records:
                f.write(json.dumps(record) + "\n")
        print(f"[{name}] {errors} failed downloads logged to {failed_log_path}")
    else:
        # Remove stale failure log if everything succeeded
        if os.path.exists(failed_log_path):
            os.remove(failed_log_path)

    print(f"[{name}] Done. Downloaded={ok}, Skipped={skipped}, Errors={errors}")
    return ok, skipped, errors


def main():
    total_ok = total_skipped = total_errors = 0
    for dataset in DATASETS:
        ok, skipped, errors = process_dataset(dataset)
        total_ok += ok
        total_skipped += skipped
        total_errors += errors

    print(f"\n=== All datasets complete ===")
    print(f"Total Downloaded={total_ok}, Skipped={total_skipped}, Errors={total_errors}")


if __name__ == "__main__":
    main()
