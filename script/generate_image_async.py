import glob
import json
import os
import time
import urllib.request
from typing import Any, Dict

import cv2
import numpy as np
import websockets.client as wc

INPUT_PREFIX = "INPUT_"
INPUT_PREFIX_LEN = len(INPUT_PREFIX)

OUTPUT_PREFIX = "OUT_"
OUTPUT_PREFIX_LEN = len(OUTPUT_PREFIX)

NC_LOADIMAGE = "LoadImage"
NC_SAVEIMWS = "SaveImageWebsocket"


def queue_prompt(prompt, server_address, client_id):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())


async def get_images(ws : wc.WebSocketClientProtocol, server_address : str, client_id : str, prompt : Any, out_idx : Dict[str, str]):
    prompt_id = queue_prompt(prompt, server_address, client_id)['prompt_id']
    output_images = {}
    current_node = ""
    while True:
        out = await ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['prompt_id'] == prompt_id:
                    if data['node'] is None:
                        break #Execution is done
                    else:
                        current_node = data['node']
        else:
            if current_node in out_idx:
                images_output = output_images.get(current_node, [])
                images_output.append(out[8:])
                output_images[out_idx[current_node]] = images_output

    return output_images


def make_timer():
    st = time.time()
    def _timer():
        nonlocal st
        ct = time.time()
        etime = f"{(ct-st)*1000:0.2f}"
        st = ct
        return f"{etime:12s} (ms)"

    return _timer


async def proc_image_persist(
        ws : wc.WebSocketClientProtocol,
        server_address : str,
        client_id : str,
        prompt : Any,
        param : Dict[int, tuple[str, str]],
        out_idx : Dict[str, str],
        im_path : str,
        im_out : str):

    timer = make_timer()
    print(f"[+{timer()}] Sending new query from {im_path}.")
    prompt[param[0][0]]["inputs"]["image"] = im_path

    images = await get_images(ws, server_address, client_id, prompt, out_idx)

    print(f"[+{timer()}] Received {len(images)} image(s) from the server.")

    node_suffs = list(images.keys())
    node_suffs.sort()
    rtn = []
    for node_suff in node_suffs:
        for image_data in images[node_suff]:
            out_file = f"{im_out}_{node_suff}.png"
            arr = np.asarray(bytearray(image_data))
            im = cv2.imdecode(arr, -1)
            cv2.imwrite(out_file, im)
            rtn.append(im)

    print(f"[+{timer()}] Finished writing to disk.")
    return rtn


def prompt_load(wf : str) -> Any :
    with open(wf, "r", encoding="utf-8") as fd:
        prompt = json.load(fd)
        
    return prompt



def clear_video_dir(vid_dir):
    pngs = glob.glob(vid_dir + "/**/**.png")
    mp4s = glob.glob(vid_dir + "/**/**.mp4")

    for f in pngs + mp4s:
        os.remove(f)
    
def find_io_nodes(prompt : Any) -> tuple[Dict[int, tuple[str, str]], Dict[str, str], bool] :
    input_idx_map = {}
    output_idx = {}
    for nid, val in prompt.items():
        title = val["_meta"]["title"]
        node_class = val["class_type"]
        if INPUT_PREFIX in title and title[:INPUT_PREFIX_LEN] == INPUT_PREFIX and title[INPUT_PREFIX_LEN:].isdigit():
            idx = int(title[INPUT_PREFIX_LEN:])
            input_idx_map[idx] = (nid, node_class)

        if node_class == NC_SAVEIMWS:
            suff = title[OUTPUT_PREFIX_LEN:] if len(title) > OUTPUT_PREFIX_LEN else title
            output_idx[nid] = suff


    succ = False
    if len(output_idx) == 0:
        print(f"No output node detected.")
        
    if p := 0 not in input_idx_map:
        print(f"Input node {p} not found.")
        print(f"No node with title matching {INPUT_PREFIX}{p}")
        succ = False

    if input_idx_map[0][1] != NC_LOADIMAGE:
        print(f"Input node {0} is of type \"{input_idx_map[0][1]}\". Expected type \"{NC_LOADIMAGE}\".")
        succ = False

    return input_idx_map, output_idx, succ

