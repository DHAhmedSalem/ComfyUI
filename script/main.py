import asyncio
import glob
import os
import random
import uuid
from typing import Any, Dict, List, Optional

import cv2
import gradio as gr
import numpy as np
import websockets.client as wsc
import websockets.exceptions as wse

import generate_image as gis
from generate_image_async import (find_io_nodes, make_timer,
                                  proc_image_persist, prompt_load)

workflow_file = "./template/workflow_api.json"
vid_dir = "../output/vid"
out_dir = "./out"
vid_path = os.path.join(vid_dir, "vid_00001.mp4")

# TODO: replace with proper call
def scanner():
    files = glob.glob("./tex/*.jpg")
    return os.path.abspath(files[random.randint(0, len(files) - 1)]), True

class ComfyRequest:
    def __init__(self, prompt : Any, param: Dict[int, tuple[str, str]], out_idx : Dict[str, str], im_path : str, im_prefix):
        self.prompt = prompt
        self.param = param
        self.out_idx = out_idx
        self.im_path = im_path
        self.im_prefix = im_prefix
        

class ComfyResponse:
    def __init__(self, source : str, images : List[np.ndarray], vidpath : str):
        self.source = source
        self.images = images
        self.vidpath = vidpath

    def update(self, source, images : List[np.ndarray], vidpath : str):
        self.source = source
        self.images = images
        self.vidpath = vidpath

    def get(self):
        if len(self.images) > 0 and os.path.isfile(self.vidpath) :
            return self.source, cv2.cvtColor(self.images[0], cv2.COLOR_BGRA2RGBA), self.vidpath, True

        return self.source, np.zeros((512,512,3), np.uint8), self.vidpath, False        
    

class Connection:
    def __init__(self):
        self.ws : Optional[wsc.WebSocketClientProtocol]
        self.connected = False
        self.qin : asyncio.Queue[Optional[ComfyRequest]] = asyncio.Queue()
        self.qout : asyncio.Queue[ComfyResponse] = asyncio.Queue()
        self.client_id = str(uuid.uuid4())
        
    def is_connected(self):
        return self.connected

    async def listen(self, server_address):        
        while (req := await self.qin.get()) is not None:
            try:
                if self.ws is None:
                    self.ws = await wsc.connect(f"ws://{server_address}/ws?clientId={self.client_id}")
                                
                images = await proc_image_persist(self.ws, server_address, self.client_id, req.prompt, req.param, req.out_idx, req.im_path, req.im_prefix)
                await self.qout.put(ComfyResponse(req.im_path, images, vid_path))

            except wse.WebSocketException as e:
                print(f"WebSocket exception encountered in websocket client:\n\t{e}")
                if self.ws is not None:
                    await self.ws.close()
                    self.ws = None
                
            except Exception as e:
                print(f"Exception encountered in websocket client:\n\t{e}")
                
    def stop(self):
        asyncio.run(self.qin.put(None))
            

class Workflow:
    def __init__(self):
        self.wf = ""
        self.prompt = {}
        self.param = {}
        self.out_idx = {}
        self.succ = False        

    def update(self, wf : str) -> bool:
        self.wf = wf        
        try:
            self.prompt = prompt_load(wf)            
            self.param, self.out_idx, self.succ = find_io_nodes(self.prompt)
            return self.succ
        except:
            return False

        
def app():
    timer = make_timer()
    wf = Workflow()
    wf.update(workflow_file)
    print(f"[+{timer()}] Loaded workflow file")

    ws = gis.connect_server()
    cr = ComfyResponse("", [], vid_path)

    def ensure_connected():
        nonlocal ws
        try:
            ws.ping()
        except wse.ConnectionClosed:
            ws = gis.connect_server()
        except RuntimeError:
            ws.close()
            ws = gis.connect_server()

    def generate_image_command():
        timer = make_timer()
        ensure_connected()
        print(f"[+{timer()}] Completed connection check")        
        gis.clear_video_dir(vid_dir)
        print(f"[+{timer()}] Cleared video dir") 
        fn, res = scanner()
        print(f"[+{timer()}] Received file from scanner") 
        if not res:
            return cr.get()

        out_p = os.path.join(out_dir, os.path.splitext(os.path.basename(fn))[0])
        images = gis.proc_image_persist(ws, wf.prompt, wf.param, wf.out_idx, fn, out_p)        
        if len(images) < 1:
            return cr.get()

        cr.update(fn, images, vid_path)
        return cr.get()
    
    
    with gr.Blocks(css="footer {visibility: hidden}") as inf:
        with gr.Row():
            gr.Markdown(
                """
                # "Naturalis"
                """)
        with gr.Row():
            scan_btn = gr.Button(value = "Scan")
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Evolution")
                vid_view = gr.Video(label="Evolution")
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Your Sketch")
                im_og = gr.Image(label="Your Sketch")
            with gr.Column():
                gr.Markdown("## The Morph")
                im_view = gr.Image(label="Morph")

        scan_btn.click(generate_image_command, [], [im_og, im_view, vid_view])

    inf.launch(server_name="127.0.0.1")

if __name__ == "__main__":
    app()
