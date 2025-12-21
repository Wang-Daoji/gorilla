import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from bfcl_eval.constants.eval_config import (
    PROJECT_ROOT,
    RESULT_PATH,
    TEST_IDS_TO_GENERATE_PATH,
)
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.eval_checker.eval_runner_helper import load_file
from bfcl_eval.utils import *
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    # Refer to test_categories for supported categories.
    parser.add_argument("--test-category", type=str, default=["single_turn", "multi_turn"], nargs="+")
    parser.add_argument("--result-dir", default=None, type=str)
    parser.add_argument("--run-ids", action="store_true", default=False)
    parser.add_argument("--allow-overwrite", "-o", action="store_true", default=False)
    parser.add_argument("--num-threads", default=10, type=int, help="Number of threads for parallel search")
    args = parser.parse_args()
    return args


def get_client() -> Any:
    """Initialize the memory client based on FRAME environment variable."""
    if os.getenv("FRAME") == "memobase":
        from utils.client import MemobaseClient
        return MemobaseClient()
    elif os.getenv("FRAME") == "memos-api":
        from utils.client import MemosApiClient
        return MemosApiClient()
    elif os.getenv("FRAME") == "memos-api-online":
        from utils.client import MemosApiOnlineClient
        return MemosApiOnlineClient()
    elif os.getenv("FRAME") == "memu":
        from utils.client import MemuClient
        return MemuClient()
    elif os.getenv("FRAME") == "supermemory":
        from utils.client import SupermemoryClient
        return SupermemoryClient()
    elif os.getenv("FRAME") == "mem0":
        from utils.client import Mem0Client
        return Mem0Client()
    else:
        raise ValueError(f"Invalid frame: {os.getenv('FRAME')}")


def search_memory(client: Any, query: str, test_entry_id: str, top_k: int) -> list:
    """Search memory using the provided client."""
    version = os.getenv("VERSION", "default_version")
    frame = os.getenv("FRAME")
    user_id = f"{test_entry_id}_{frame}_{version}"
    search_rsps = client.search(query=query, user_id=user_id, top_k=top_k)

    if frame == "memos-api":
        memories = [mem for mem in search_rsps["tool_mem"][0]["memories"] if mem["metadata"]["memory_type"] == "ToolTrajectoryMemory"]
    elif frame == "mem0":
        memories = search_rsps["results"]
    elif frame == "supermemory":
        memories = search_rsps.results
    else:
        raise ValueError(f"Invalid frame: {frame}")

    return memories


def create_mem_context(memories: list[dict] | str | dict) -> str:
    """Create memory context from search results."""
    frame = os.getenv("FRAME")
    if frame == "memos-api":
        text_mem_context = ""
        tool_trajectory_memories = [
            item
            for item in memories
            if item["metadata"]["memory_type"] == "ToolTrajectoryMemory"
        ]
        tool_mem_context = "\n".join(
            [
                f"{i + 1}. tool_trajectory: {item['memory']}\nexperience: {item['metadata']['experience']}\ntool_used_status: {json.dumps(item['metadata']['tool_used_status'], ensure_ascii=False)}"
                for i, item in enumerate(tool_trajectory_memories)
            ]
        )

        if text_mem_context:
            text_mem_context = "Fact Memory:\n" + text_mem_context
        if tool_mem_context:
            tool_mem_context = "Tool Memory:\n" + tool_mem_context

        mem_context = text_mem_context + "\n" + tool_mem_context
    elif frame == "mem0":
        mem_context = "\n".join([f"{memory['created_at']}: {memory['memory']}" for memory in memories])
    elif frame == "supermemory":
        mem_context = "\n".join([memory["memory"] for memory in memories])
    else:
        raise ValueError(f"Invalid frame: {frame}")

    return mem_context.strip()


def get_involved_test_entries(test_category_args, run_ids):
    """Load test entries based on category or ID file."""
    all_test_categories, all_test_entries_involved = [], []
    if run_ids:
        all_test_categories, all_test_entries_involved = load_test_entries_from_id_file(
            TEST_IDS_TO_GENERATE_PATH
        )
    else:
        all_test_categories = parse_test_category_argument(test_category_args)
        for test_category in all_test_categories:
            all_test_entries_involved.extend(load_dataset_entry(test_category))

    return all_test_categories, all_test_entries_involved


def collect_test_cases(args, all_test_entries_involved):
    """Collect test cases that need search results."""
    result_dir = args.result_dir if args.result_dir is not None else RESULT_PATH
    result_dir = Path(result_dir)
    
    search_result_file = result_dir / "search_results.jsonl"
    
    existing_ids = set()
    if search_result_file.exists() and not args.allow_overwrite:
        with open(search_result_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                existing_ids.add(entry["id"])
    elif search_result_file.exists() and args.allow_overwrite and not args.run_ids:
        # Delete existing file if overwrite is allowed and not running specific IDs
        search_result_file.unlink()
        existing_ids = set()

    test_cases_to_search = [
        test_case
        for test_case in all_test_entries_involved
        if test_case["id"] not in existing_ids
    ]
    
    return sorted(test_cases_to_search, key=sort_key)


def search_single_test_case(client: Any, test_case: dict, top_k: int = 10) -> dict:
    """Search memory for a single test case."""
    test_entry_id = test_case["id"]
    
    # Extract all user questions from the test case
    all_questions = []
    if "question" in test_case:
        for turn in test_case["question"]:
            for msg in turn:
                if msg["role"] == "user":
                    all_questions.append(msg["content"])
    
    # Perform searches for all questions
    mem_list = []
    for question in all_questions:
        try:
            memories = search_memory(client, question, test_entry_id, top_k)
            mem_list.extend(memories)
        except Exception as e:
            tqdm.write(f"Error searching for test case {test_entry_id}, question: {question[:50]}... Error: {str(e)}")
            continue
    
    # Deduplicate memories by id field
    seen_ids = set()
    dedup_mem_list = []
    for item in mem_list:
        # Convert class instance to dict if needed
        if not isinstance(item, dict):
            item = item.__dict__ if hasattr(item, '__dict__') else dict(item)
        item_id = item.get("id")
        if item_id not in seen_ids:
            seen_ids.add(item_id)
            dedup_mem_list.append(item)
    
    # Create memory context
    mem_context = ""
    if dedup_mem_list:
        try:
            mem_context = create_mem_context(dedup_mem_list)
        except Exception as e:
            tqdm.write(f"Error creating memory context for test case {test_entry_id}: {str(e)}")
    
    # Return search result
    result = {
        "id": test_entry_id,
        "questions": all_questions,
        "memories": dedup_mem_list,
        "mem_context": mem_context,
    }
    
    return result


def generate_search_results(args, test_cases_total):
    """Generate and save search results for all test cases."""
    result_dir = args.result_dir if args.result_dir is not None else RESULT_PATH
    result_dir = Path(result_dir)
    
    search_result_file = result_dir / "search_results.jsonl"
    
    # Initialize memory client
    client = get_client()
    
    # Use ThreadPoolExecutor for parallel search
    num_threads = args.num_threads
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor, \
         open(search_result_file, "a") as f, \
         tqdm(total=len(test_cases_total), desc="Generating search results") as pbar:
        
        # Submit all tasks
        futures = {
            executor.submit(search_single_test_case, client, test_case): test_case
            for test_case in test_cases_total
        }
        
        # Process completed tasks
        for future in as_completed(futures):
            try:
                result = future.result()
                # Write result to file
                json_line = json.dumps(result, ensure_ascii=False) + "\n"
                f.write(json_line)
                f.flush()
                pbar.update(1)
            except Exception as e:
                test_case = futures[future]
                tqdm.write(f"Error processing test case {test_case['id']}: {str(e)}")
                pbar.update(1)


def main(args):
    if type(args.test_category) is not list:
        args.test_category = [args.test_category]
    
    (
        all_test_categories,
        all_test_entries_involved,
    ) = get_involved_test_entries(args.test_category, args.run_ids)
    
    tqdm.write(f"Generating search results for categories: {all_test_categories}")
    
    if args.result_dir is not None:
        args.result_dir = Path(args.result_dir).resolve()
    else:
        args.result_dir = RESULT_PATH
    
    test_cases_total = collect_test_cases(args, all_test_entries_involved)
    
    if len(test_cases_total) == 0:
        tqdm.write(
            "✅ All selected test cases already have search results. No new search to generate."
        )
    else:
        generate_search_results(args, test_cases_total)
        tqdm.write(f"✅ Generated search results for {len(test_cases_total)} test cases.")


if __name__ == "__main__":
    args = get_args()
    main(args)
