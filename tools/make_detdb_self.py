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


det_db = {}
to_cache = []
parent_path = "/home/kelaboss/eval_data/xcyuan"
path_list = os.listdir(parent_path)
for path in path_list:
    vid = os.path.join(parent_path, path)
    vid = Path(vid)
    if vid.is_dir():
        file_path = os.path.join(vid, "deoverlap_yolov5_yolox/exp/new")
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

with open("/home/kelaboss/eval_data/xcyuan/det_db_oc_sort_full.json", 'w') as f:
    json.dump(det_db, f)
