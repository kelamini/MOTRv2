# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


import os
from pathlib import Path
from glob import glob
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from tqdm import tqdm


def multi_folder():
    det_db = {}
    to_cache = []
    parent_path = "/home/kelaboss/eval_data/xcyuan/person_data_clean"
    path_list = sorted(os.listdir(parent_path))
    for paths in path_list:
        vid = os.path.join(parent_path, paths)
        vid = Path(vid)
        
        if vid.is_dir() and paths != ".git":
            print(vid)
            file_path = os.path.join(vid, "multi_yolox_8x8_300e_coco/exp/vote_new")
            for file in glob(os.path.join(file_path, "*.json")):
                to_cache.append(file)


    pbar = tqdm(total=len(to_cache))

    mutex = Lock()
    def cache(file):
        with open(file, "r") as fp:
            data = json.load(fp)
        bbox_list = []
        for shape in data["shapes"]:
            points = shape["points"]
            score = shape["score"]
            l, t, w, h, s = points[0][0], points[0][1], points[1][0]-points[0][0], points[1][1]-points[0][1], score
            bbox_list.append([l, t, w, h, s])
        
        with mutex:
            det_db[file] = bbox_list
            pbar.update()

    with ThreadPoolExecutor(max_workers=48) as exe:
        for file in to_cache:
            exe.submit(cache, file)

    with open("/home/kelaboss/eval_data/xcyuan/person_data_clean/det_db_person.json", 'w') as f:
        json.dump(det_db, f, indent=2)


def only_folder():
    det_db = {}
    to_cache = []
    parent_path = "/home/kelaboss/eval_data/xcyuan/person_data_clean/172.25.188.24"
    vid = Path(parent_path)
    if vid.is_dir():
        file_path = os.path.join(vid, "multi_yolox_8x8_300e_coco/exp/vote_new")
        for file in glob(os.path.join(file_path, "*.json")):
            to_cache.append(file)


    pbar = tqdm(total=len(to_cache))

    mutex = Lock()
    def cache(file):
        with open(file, "r") as fp:
            data = json.load(fp)
        bbox_list = []
        for shape in data["shapes"]:
            points = shape["points"]
            score = shape["score"]
            l, t, w, h, s = points[0][0], points[0][1], points[1][0]-points[0][0], points[1][1]-points[0][1], score
            bbox_list.append([l, t, w, h, s])
        
        with mutex:
            det_db[file] = bbox_list
            pbar.update()

    with ThreadPoolExecutor(max_workers=48) as exe:
        for file in to_cache:
            exe.submit(cache, file)

    with open("/home/kelaboss/eval_data/xcyuan/person_data_clean/det_db_person.json", 'w') as f:
        json.dump(det_db, f, indent=2)


if __name__ == "__main__":
    
    # only_folder()

    multi_folder()
