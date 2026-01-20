import os
import pickle
import shutil
import multiprocessing
import time
from util.pairdata import pairdata
from util.path_utils import mkdir_of_file, get_save_path, get_ida_log_path
import argparse
from tqdm import tqdm

from my_config import wsl_to_win_path, run_ida

from my_config import (
    IDA_PATH,
    BASE_PATH,
    STRIP_PATH,
    IDA_SCRIPT_PATH,
    DATA_ROOT,
    SAVE_ROOT,
    NUM_JOBS,
    LOG_PATH,
    IDB_PATH,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-rate", type=float, default=0.5)
    args = parser.parse_args()

    ida_log_dir = LOG_PATH
    ida_idb_dir = IDB_PATH

    print(f"[*] IDA_PATH:     {IDA_PATH}")
    print(f"[*] BASE_PATH:    {BASE_PATH}")
    print(f"[*] BASE_PATH:    {BASE_PATH}")
    print(f"[*] DATA_ROOT:    {DATA_ROOT}")
    print(f"[*] STRIP_PATH:   {STRIP_PATH}")
    print(f"[*] SAVE_ROOT:    {SAVE_ROOT}")
    print(f"[*] LOG_PATH:     {LOG_PATH}")
    print(f"[*] IDB_PATH:     {IDB_PATH}")
    print()

    os.makedirs(STRIP_PATH, exist_ok=True)
    os.makedirs(IDB_PATH,   exist_ok=True)
    os.makedirs(LOG_PATH,   exist_ok=True)
    os.makedirs(SAVE_ROOT, exist_ok=True)

    start = time.time()
    bin_pathes = sorted([os.path.join(root, file) for root, dirs, files in os.walk(DATA_ROOT) for file in files])

    #! TODO: Fix me
    bin_pathes = bin_pathes[:2]

    jobs = []
    # pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count() * args.cpu_rate))
    for bin_path in bin_pathes:
        print(f"[*] bin_path: {bin_path}")

        bin_dir, bin_name = os.path.split(bin_path)
        print(f"[*] bin_dir:  {bin_dir}")
        print(f"[*] bin_name: {bin_name}")
        assert bin_dir.startswith(DATA_ROOT), (bin_dir, DATA_ROOT)

        # bin_dir_suffix = bin_dir[len(DATA_ROOT):]
        bin_dir_suffix = ""

        strip_bin_path = os.path.join(STRIP_PATH + bin_dir_suffix, f"{bin_name}.strip")
        print(f"[*] strip_bin_path: {strip_bin_path}")
        

        if not os.path.exists(strip_bin_path):
            print(os.path.split(strip_bin_path)[0])
            mkdir_of_file(strip_bin_path)
            cmd = f"strip -s {bin_path} -o {strip_bin_path}"
            print("[jTrans.run]", cmd)
            os.system(cmd)

            #! ???
            if not os.path.exists(strip_bin_path):  # if not x86 elf
                os.makedirs(os.path.split(strip_bin_path)[0], exist_ok=True)
                shutil.copy(bin_path, strip_bin_path)


        save_path = get_save_path(SAVE_ROOT, bin_dir_suffix, bin_name)
        if os.path.exists(save_path):
            try:
                res = pickle.load(open(save_path, 'rb'))
                continue
            except EOFError as e:  # wrong pkl file
                pass

        print("[jTrans.run] to obtain", save_path)

        log_path = get_ida_log_path(ida_log_dir, bin_dir_suffix, bin_name)
        idb_path = os.path.join(ida_idb_dir + bin_dir_suffix, f'{bin_name}.idb')
        mkdir_of_file(log_path)
        mkdir_of_file(idb_path)


        log_path_win = wsl_to_win_path(log_path)
        idb_path_win = wsl_to_win_path(idb_path)
        strip_bin_path_win = wsl_to_win_path(strip_bin_path)

        cmd_win = [IDA_PATH, f'-L{log_path_win}', '-c', '-A', f'-S{IDA_SCRIPT_PATH}', f'-o{idb_path_win}', f'{strip_bin_path_win}']
        print(" ".join(cmd_win))

        out_path = os.path.join(f"{log_path}.ida.stdout.txt")
        err_path = os.path.join(f"{log_path}.ida.stderr.txt")

        jobs.append((cmd_win, out_path, err_path, True))

        # os.system(' '.join(cmd_win))
        print("[*] IDA processing..")
        with multiprocessing.Pool(processes=NUM_JOBS) as pool:
            for _ in tqdm(
                pool.imap_unordered(run_ida, jobs),
                total=len(jobs),
            ):
                pass

        # pool.apply_async(subprocess.call, args=(cmd,))
    # pool.close()
    # pool.join()

    print("[*] Check..")

    # check whether generate .pkl is exists
    for bin_path in bin_pathes:
        bin_dir, bin_name = os.path.split(bin_path)
        # bin_dir_suffix = bin_dir[len(DATA_ROOT):]
        bin_dir_suffix = ""
        save_path = get_save_path(SAVE_ROOT, bin_dir_suffix, bin_name)
        if not os.path.exists(save_path):
            log_path = get_ida_log_path(ida_log_dir, bin_dir_suffix, bin_name)
            print(  f"[jTrans.run] XXXXXXXXXXXXXXX fail extract ida feat of {bin_path} into {save_path},\n"
                    f"see log {log_path} for details.")

    print('[*] Features Extracting Done')
    # pairdata(SAVE_ROOT)
    end = time.time()
    print(f"[*] Time Cost: {end - start} seconds")

if __name__ == '__main__':
    main()

# EOF
