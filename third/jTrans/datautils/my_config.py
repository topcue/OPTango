import os
import subprocess

IDA_PATH = "/home/user/win_workspace/IDA/idat64.exe"

BASE_PATH = "/home/user/win_workspace/storage/optango"

DATA_ROOT = os.path.join(BASE_PATH, "dataset")
STRIP_PATH = os.path.join(BASE_PATH, "dataset_strip")
LOG_PATH = os.path.join(BASE_PATH, "log")
IDB_PATH = os.path.join(BASE_PATH, "idb")
SAVE_ROOT = os.path.join(BASE_PATH, "extract")

LOG_PATH = os.path.join(BASE_PATH, "log")
IDB_PATH = os.path.join(BASE_PATH, "idb")

cur_script_dir_path = os.path.dirname(os.path.abspath(__file__))
IDA_SCRIPT_PATH = os.path.join(cur_script_dir_path, "process.py")

NUM_JOBS = 24

#! $ mkdir -p /mnt/c/Users/user/workspace
#! $ ln -s /mnt/c/Users/user/workspace /home/user/win_workspace
WSL_PREFIX = "/home/user/win_workspace"
WIN_PREFIX = "C:/Users/user/workspace"

def wsl_to_win_path(p):
    if p == WSL_PREFIX or p.startswith(WSL_PREFIX + "/"):
        return WIN_PREFIX + p[len(WSL_PREFIX):]
    return p

def win_to_wsl_path(p):
    if p == WIN_PREFIX or p.startswith(WIN_PREFIX + "/"):
        return WSL_PREFIX + p[len(WIN_PREFIX):]
    return p


def run_ida(args):
    cmd, out_path, err_path, debug = args
    if debug:
        with open(out_path, "wb") as out_f, open(err_path, "wb") as err_f:
            return subprocess.call(cmd, stdout=out_f, stderr=err_f)
    else:
        # Discard stdout/stderr
        return subprocess.call(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


# EOF
