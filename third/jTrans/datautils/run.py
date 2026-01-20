import os
import time
import subprocess
import multiprocessing
from tqdm import tqdm
from util.pairdata import pairdata

from my_config import wsl_to_win_path, run_ida
from my_config import (
    IDA_PATH,
    STRIP_PATH,
    IDA_SCRIPT_PATH,
    DATA_ROOT,
    SAVE_ROOT,
    NUM_JOBS,
    LOG_PATH,
    IDB_PATH,
)


def getTarget(path):
    target = []
    for root, _, files in os.walk(path):
        for file in files:
            target.append(os.path.join(root, file))
    return target


def strip_one(target_path: str) -> tuple[str, str, int]:
    filename = os.path.basename(target_path)
    stripped_path = os.path.join(STRIP_PATH, filename + ".strip")

    # Use subprocess to get a real return code
    # strip -s <in> -o <out>
    cmd_strip = f"strip -s {target_path} -o {stripped_path}"
    ret = subprocess.call(cmd_strip.split(' '))
    return (target_path, stripped_path, ret)


def strip_all_parallel(target_list, num_jobs: int):
    results = []
    with multiprocessing.Pool(processes=num_jobs) as pool:
        for orig, stripped, ret in tqdm(
            pool.imap_unordered(strip_one, target_list, chunksize=16),
            total=len(target_list),
        ):
            if ret != 0:
                print(f"[WARN] strip failed: {orig} (ret={ret})")
                continue
            results.append(stripped)
    return results


def main():
    DEBUG = False

    os.makedirs(STRIP_PATH, exist_ok=True)
    os.makedirs(IDB_PATH,   exist_ok=True)
    os.makedirs(LOG_PATH,   exist_ok=True)
    os.makedirs(SAVE_ROOT, exist_ok=True)

    start = time.time()

    target_list = getTarget(DATA_ROOT)

    # Stage 1) strip in parallel
    print("[*] Stripping binaries..")
    stripped_list = strip_all_parallel(target_list, num_jobs=NUM_JOBS)

    # Stage 2) run IDA in parallel on stripped files
    jobs = []
    for ida_input in stripped_list:
        filename_strip = os.path.basename(ida_input)          # e.g., xxx.strip
        filename = filename_strip.replace(".strip", "")       # e.g., xxx

        ida_input_win = wsl_to_win_path(ida_input)
        IDB_PATH_WIN = wsl_to_win_path(IDB_PATH)
        LOG_PATH_WIN = wsl_to_win_path(LOG_PATH)

        cmd_str = (
            f"{IDA_PATH} -L{LOG_PATH_WIN}/{filename}.log "
            f"-c -A -S{IDA_SCRIPT_PATH} "
            f"-o{IDB_PATH_WIN}/{filename}.idb {ida_input_win}"
        )
        cmd = cmd_str.split(" ")

        out_path = os.path.join(LOG_PATH, f"{filename}.ida.stdout.txt")
        err_path = os.path.join(LOG_PATH, f"{filename}.ida.stderr.txt")

        jobs.append((cmd, out_path, err_path, DEBUG))

    print("[*] IDA processing..")
    with multiprocessing.Pool(processes=NUM_JOBS) as pool:
        for _ in tqdm(
            pool.imap_unordered(run_ida, jobs),
            total=len(jobs),
        ):
            pass

    print("[*] Features Extracting Done")
    print("[*] Saving..")
    pairdata(SAVE_ROOT)

    end = time.time()
    print(f"[*] Time Cost: {end - start} seconds")


if __name__ == "__main__":
    main()

# EOF
