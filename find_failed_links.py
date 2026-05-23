import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

JSONL_PATH = "/work/u1304848/AI/project/datasets/meta_All_Beauty.jsonl"
OUTPUT_PATH = "/work/u1304848/AI/project/failed_image_links.jsonl"
TARGET_FAILURES = 50
MAX_WORKERS = 32


def check_url(task):
    url, record, variant = task
    try:
        response = requests.head(url, timeout=10, allow_redirects=True)
        if response.status_code >= 400:
            return {"url": url, "status_code": response.status_code, "record": record, "variant": variant}
    except Exception as e:
        return {"url": url, "error": str(e), "record": record, "variant": variant}
    return None


def collect_tasks():
    tasks = []
    with open(JSONL_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for img in record.get("images", []):
                url = img.get("large")
                variant = img.get("variant")
                if url and variant and record.get("parent_asin"):
                    tasks.append((url, record, variant))
    return tasks


def main():
    print("Collecting image tasks...")
    tasks = collect_tasks()
    print(f"Total image URLs to check: {len(tasks)}")

    failures = []
    checked = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(check_url, task): task for task in tasks}
        with tqdm(total=len(tasks), unit="url") as bar:
            for future in as_completed(futures):
                checked += 1
                result = future.result()
                if result is not None:
                    record = result.pop("record")
                    entry = {
                        "parent_asin": record.get("parent_asin"),
                        "title": record.get("title"),
                        "main_category": record.get("main_category"),
                        "price": record.get("price"),
                        "average_rating": record.get("average_rating"),
                        "rating_number": record.get("rating_number"),
                        "store": record.get("store"),
                        "failed_image_url": result["url"],
                        "variant": result["variant"],
                    }
                    if "status_code" in result:
                        entry["failure_reason"] = f"HTTP {result['status_code']}"
                    else:
                        entry["failure_reason"] = result.get("error", "unknown")
                    failures.append(entry)
                    tqdm.write(f"  FAILED [{len(failures)}] {entry['parent_asin']} {entry['variant']}: {entry['failure_reason']}")

                bar.set_postfix(checked=checked, failed=len(failures))
                bar.update(1)

                if len(failures) >= TARGET_FAILURES:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break

    failures = failures[:TARGET_FAILURES]
    with open(OUTPUT_PATH, "w") as f:
        for entry in failures:
            f.write(json.dumps(entry) + "\n")

    print(f"\nDone. Found {len(failures)} failed links. Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
