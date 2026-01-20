import os
from collections import defaultdict
from tqdm import tqdm
from shutil import move
import pickle


#! TODO: Fix to fit BinForge
def get_prefix_from_name(filename: str) -> str:
    # filename example:
    #   libcap-git-setcap-O0-<hash>_extract.pkl
    # Remove the last two tokens (e.g., O0, <hash>_extract.pkl)
    # Resulting prefix:
    #   libcap-git-setcap
    parts = filename.split('-')
    prefix = '-'.join(parts[:-2])
    return prefix


def pairdata(data_dir):
    # Mapping: project name -> list of full pickle paths
    proj2paths = defaultdict(list)

    # 1) Collect input pickle files
    #    - Only consider *_extract.pkl
    #    - Explicitly exclude saved_index.pkl
    for root, _, files in os.walk(data_dir):
        for name in files:
            if not name.endswith("_extract.pkl"):
                continue
            prefix = get_prefix_from_name(name)
            proj2paths[prefix].append(os.path.join(root, name))

    # 2) Process each project independently
    for proj, paths in proj2paths.items():
        proj_dir = os.path.join(data_dir, proj)
        os.makedirs(proj_dir, exist_ok=True)

        binary_func_list = []  # list of function-name lists per pickle
        pkl_list = []          # list of loaded pickle objects

        print(f"[*] proj: {proj}")
        for src_path in tqdm(paths):
            name = os.path.basename(src_path)
            dst_path = os.path.join(proj_dir, name)

            # (a) Load pickle file
            with open(src_path, "rb") as f:
                pkl = pickle.load(f)
            pkl_list.append(pkl)

            # Collect function names from this pickle
            func_list = list(pkl.keys())
            # print(name, len(func_list))
            binary_func_list.append(func_list)

            # (b) Organize files:
            #     - If the file is outside proj_dir, move it inside
            #     - If a file with the same name already exists, overwrite it
            if os.path.abspath(src_path) != os.path.abspath(dst_path):
                if os.path.exists(dst_path):
                    os.remove(dst_path)
                move(src_path, dst_path)

        # 3) Compute intersection of function names
        #    If there is only one pickle, the intersection is the full set
        if not binary_func_list:
            continue
        final_index = set(binary_func_list[0])
        for lst in binary_func_list[1:]:
            final_index &= set(lst)

        # print("all", len(final_index))

        # 4) Build paired data:
        #    func_name -> [value_from_pkl1, value_from_pkl2, ...]
        saved_index = defaultdict(list)
        for func_name in final_index:
            for pkl in pkl_list:
                saved_index[func_name].append(pkl[func_name])

        # 5) Always overwrite the output pickle
        saved_pickle_path = os.path.join(proj_dir, "saved_index.pkl")
        with open(saved_pickle_path, "wb") as f:
            pickle.dump(dict(saved_index), f)

# EOF
