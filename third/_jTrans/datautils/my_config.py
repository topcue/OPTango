import sys

PROJECT_NAME = "optango"

def setup_ida_sys_path():
    # Add any extra paths needed by IDA python (example: your site-packages)
    ida_site_pkgs = f"C:\\Users\\user\\workspace\\IDA-python\\{PROJECT_NAME}\\Lib\\site-packages"
    if ida_site_pkgs not in sys.path:
        sys.path.insert(0, ida_site_pkgs)
    
    def p(*a):
        print("[IDAPY]", *a)

    p("==== sys.path ====")
    for i, x in enumerate(sys.path):
        p(i, x)

import os
import subprocess

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

IDA_PATH = "/home/user/win_workspace/IDA/idat64.exe"

BASE_PATH = f"/home/user/win_workspace/storage/{PROJECT_NAME}"

DATA_ROOT = os.path.join(BASE_PATH, "dataset")
STRIP_PATH = os.path.join(BASE_PATH, "dataset_strip")
LOG_PATH = os.path.join(BASE_PATH, "log")
IDB_PATH = os.path.join(BASE_PATH, "idb")
SAVE_ROOT = os.path.join(BASE_PATH, "extract")

LOG_PATH = os.path.join(BASE_PATH, "log")
IDB_PATH = os.path.join(BASE_PATH, "idb")

cur_script_dir_path = os.path.dirname(os.path.abspath(__file__))
IDA_SCRIPT_PATH = os.path.join(cur_script_dir_path, "process.py")

DATA_ROOT_WIN = wsl_to_win_path(DATA_ROOT)
SAVE_ROOT_WIN = wsl_to_win_path(SAVE_ROOT)

NUM_JOBS = 24


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
