import argparse
import json
import os
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print(f"sys.path: {sys.path}")


def load_data(data_dir: str) -> list:
    """
    Load all JSONL files from the specified directory into a single list.

    Args:
        data_dir: Path to the directory containing JSONL files. If None, uses default path relative to this script.

    Returns:
        List containing all JSON objects from all JSONL files.
    """
    data = []

    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Directory {data_dir} does not exist")

    # Find all .jsonl files in the directory
    jsonl_files = list(data_path.glob("*.jsonl"))

    if not jsonl_files:
        print(f"Warning: No .jsonl files found in {data_dir}")
        return data

    # Load data from each JSONL file
    for jsonl_file in jsonl_files:
        print(f"Loading {jsonl_file.name}...")
        with open(jsonl_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        json_obj = json.loads(line)
                        data.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num} in {jsonl_file.name}: {e}")
                        continue

    print(f"Loaded {len(data)} records from {len(jsonl_files)} files")
    return data


def get_client(frame):
    if frame == "memobase":
        from utils.client import MemobaseClient

        return MemobaseClient()
    elif frame == "memos-api":
        from utils.client import MemosApiClient

        return MemosApiClient()
    elif frame == "memos-api-online":
        from utils.client import MemosApiOnlineClient

        return MemosApiOnlineClient()
    elif frame == "memu":
        from utils.client import MemuClient

        return MemuClient()
    elif frame == "supermemory":
        from utils.client import SupermemoryClient

        return SupermemoryClient()
    elif frame == "mem0":
        from utils.client import Mem0Client

        return Mem0Client()
    else:
        raise ValueError(f"Invalid frame: {frame}")


def process_single_conversation(conversation, client, frame):
    """Process a single conversation based on the FRAME environment variable."""
    version = os.getenv("VERSION", "default_version")
    user_id = f"{conversation['sample_id']}_{frame}_{version}"
    trajectory = conversation["trajectory"]

    if frame == "memobase":
        messages = []
        for _idx, msg in enumerate(trajectory):
            messages.append(
                {
                    "role": msg["role"],
                    "content": msg["content"][:8000],
                    "chat_time": datetime.now().isoformat(),
                }
            )
        client.add(messages, user_id)
    elif frame == "memos-api":
        client.add(trajectory, user_id, conv_id="")
    elif frame == "memos-api-online":
        client.add(trajectory, user_id)
    elif frame == "memu":
        client.add(trajectory, user_id, datetime.now().isoformat())
    elif frame == "supermemory":
        messages = []
        for _idx, msg in enumerate(trajectory):
            messages.append(
                {
                    "role": msg["role"],
                    "content": msg["content"][:8000],
                    "chat_time": datetime.now().isoformat(),
                }
            )
        client.add(messages, user_id)
    elif frame == "mem0":
        client.add(trajectory, user_id, int(datetime.now().timestamp()))
    else:
        raise ValueError(f"Invalid frame: {frame}")


def ingest_conversation(
    data: list, client, frame, max_workers: int = 32, success_records: list = [], f=None
):
    """
    Ingest conversations using thread pool for parallel processing.

    Args:
        data: List of conversation data
        client: Client object for ingestion
        max_workers: Maximum number of worker threads. If None, uses default (cpu_count * 5)
    """

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_conversation = {
            executor.submit(process_single_conversation, conversation, client, frame): conversation
            for conversation in data
            if conversation["sample_id"] not in success_records
        }

        print(
            f"Starting ingestion of {len(future_to_conversation)} conversations using {max_workers} workers..."
        )

        # Process completed tasks with progress tracking
        completed_count = 0
        total_to_process = len(future_to_conversation)
        with tqdm(total=total_to_process, desc="Ingesting conversations") as pbar:
            for future in as_completed(future_to_conversation):
                conversation = future_to_conversation[future]
                try:
                    future.result()  # This will raise any exceptions that occurred
                    completed_count += 1
                    if f is not None:
                        f.write(f"{conversation['sample_id']}\n")
                        f.flush()
                except Exception as e:
                    print(
                        f"Error processing conversation {conversation.get('sample_id', 'unknown')}: {e}"
                    )
                pbar.update(1)

    print(f"Successfully ingested {completed_count}/{total_to_process} conversations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Berkeley Function Call Leaderboard Ingestion Script"
    )
    parser.add_argument(
        "--lib",
        type=str,
        choices=[
            "mem0",
            "mem0_graph",
            "memos-api",
            "memos-api-online",
            "memobase",
            "memu",
            "supermemory",
        ],
        default="memos-api",
    )
    parser.add_argument(
        "--workers", type=int, default=5, help="Number of runs for LLM-as-a-Judge evaluation."
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/bfcl-v4", help="Directory of the data."
    )
    parser.add_argument(
        "--record-dir", type=str, default="results/bfcl-default", help="Directory to store the success records."
    )

    args = parser.parse_args()

    data = load_data(args.data_dir)
    client = get_client(args.lib)

    # read finsh task records
    os.makedirs(
        args.record_dir,
        exist_ok=True,
    )
    success_records = []
    record_file = f"{args.record_dir}/success_records.txt"
    if os.path.exists(record_file):
        with open(record_file) as f:
            for i in f.readlines():
                success_records.append(i.strip())

    # print the length of success_records and data total length
    print(
        f"record_file: {record_file} | record_length: {len(success_records)} | data_length: {len(data)}"
    )

    with open(record_file, "a+") as f:
        ingest_conversation(data, client, args.lib, args.workers, success_records, f)
