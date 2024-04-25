#This is an example that uses the websockets api and the SaveImageWebsocket node to get images directly without
#them being saved to disk

# import websockets #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
from typing import Any, Dict
import websockets.sync.client as wsc
import uuid
import json
import urllib.request
import urllib.parse
import numpy as np
import argparse
import os
import cv2
import time

INPUT_PREFIX = "INPUT_"
INPUT_PREFIX_LEN = len(INPUT_PREFIX)

OUTPUT_PREFIX = "OUT_"
OUTPUT_PREFIX_LEN = len(OUTPUT_PREFIX)

NC_LOADIMAGE = "LoadImage"
NC_SAVEIMWS = "SaveImageWebsocket"

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

workflow_file = "./template/workflow_api_simple.json"
workflow_file = "./template/workflow_api.json"

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt, out_idx):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    current_node = ""
    while True:
        out = ws.recv()        
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


def find_io_nodes(prompt : Any) -> tuple[Dict[int, tuple[str, str]], Dict[str, str]] :
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

    return input_idx_map, output_idx

def proc_image(wf: str, im_path : str, im_out : str, view : bool):
    st = time.time()
    def timer():
        nonlocal st
        ct = time.time()
        etime = f"{(ct-st)*1000:0.2f}"
        st = ct
        return f"{etime:12s} (ms)"
    
    with open(wf, "r", encoding="utf-8") as fd:
        prompt = json.load(fd)

    print(f"[+{timer()}] Loaded workflow file")
        
    param, out_idx = find_io_nodes(prompt)

    if len(out_idx) == 0:
        print(f"No output node detected.")
        return {}
    
    if p := 0 not in param:
        print(f"Input node {p} not found.")
        print(f"No node with title matching {INPUT_PREFIX}{p}")
        return {}

    if param[0][1] != NC_LOADIMAGE:
        print(f"Input node {0} is of type \"{param[0][1]}\". Expected type \"{NC_LOADIMAGE}\".")
        return {}

    print(f"[+{timer()}] Parsed workflow file for I/O")    
    
    prompt[param[0][0]]["inputs"]["image"] = im_path

    print(f"[+{timer()}] Finished preparing the query.")
    
    ws = wsc.connect("ws://{}/ws?clientId={}".format(server_address, client_id))

    print(f"[+{timer()}] Connected to the server.")
    
    images = get_images(ws, prompt, out_idx)

    print(f"[+{timer()}] Received {len(images)} image(s) from the server.")
    
    for node_suff in images:
        for image_data in images[node_suff]:
            out_file = f"{im_out}_{node_suff}.png"
            arr = np.asarray(bytearray(image_data))
            im = cv2.imdecode(arr, -1)
            cv2.imwrite(out_file, im)

    print(f"[+{timer()}] Finished writing to disk.")
    
            
    if view:
        win_title = "GEN_IMAGE_PREVIEW_WINDOW"
        cv2.namedWindow(win_title)
        for node_suff in images:
            for image_data in images[node_suff]:
                out_file = f"{im_out}_{node_suff}"
                arr = np.asarray(bytearray(image_data))
                im = cv2.imdecode(arr, -1)
                cv2.imshow(win_title, im)
                cv2.setWindowTitle(win_title, os.path.basename(out_file))
                cv2.waitKey(0)

        cv2.destroyWindow(win_title)
        print(f"[+{timer()}] Finished review of images.")
        
    ws.close()
    print(f"[+{timer()}] Disconnected from the pserver.")
    return images

def run():
    parser = argparse.ArgumentParser()
    prog_n = os.path.basename(__file__)
    prog = f"python {prog_n}"
    desc = "Image to image generation via the ComfyUI websocket server."
    epil = ""
    parser = argparse.ArgumentParser(prog=prog, description=desc, epilog=epil)
    parser.add_argument("IMAGE_IN", help="Path to the input image file.")
    parser.add_argument("IMAGE_OUT_PREFIX", help="Prefix path to the output image files.")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=False, help="Show generated images")
    parser.add_argument("-wf", "--workflow-file", default=workflow_file, help="Override the default workflow file")
    args = parser.parse_args()

    wf = args.workflow_file
    im_file = args.IMAGE_IN
    im_out = args.IMAGE_OUT_PREFIX
    show = args.show
    proc_image(wf, im_file, im_out, show)


if __name__ == "__main__":
    run()
