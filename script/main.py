import glob
import os
import random
import threading
import time
from typing import List

import cv2
import gradio as gr
import numpy as np
import websockets.exceptions as wse

import generate_image as gis

workflow_file = "./template/workflow_api.json"
vid_dir = "../output/vid"
out_dir = "./out"
vid_path = os.path.join(vid_dir, "vid_00001.mp4")


# nat_logo = cv2.imread("./res/naturalis_logo.png")
# dhs_logo = cv2.imread("./res/dhs_logo.png")
# nat_logo = cv2.resize(nat_logo, None, fx = 124 / nat_logo.shape[1], fy = 124 / nat_logo.shape[1])
# dhs_logo = cv2.resize(dhs_logo, None, fx = 124 / dhs_logo.shape[1], fy = 124 / dhs_logo.shape[1])


# TODO: replace with proper call
def scanner():
    files = glob.glob("./tex/*.jpg")
    return os.path.abspath(files[random.randint(0, len(files) - 1)]), True

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
            self.prompt = gis.prompt_load(wf)            
            self.param, self.out_idx = gis.find_io_nodes(self.prompt)
            self.succ = True
            return self.succ
        except:
            return False

        
def app():
    timer = gis.make_timer()
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

    def generate_image_command(progress=gr.Progress()):
        prog = 0
        done = False

        amount = 10000
        progress.tqdm(range(amount))
        def fun():
            nonlocal prog, done
            try:
                timer = gis.make_timer()
                ensure_connected()
                print(f"[+{timer()}] Completed connection check")
                prog = 0.1

                gis.clear_video_dir(vid_dir)
                print(f"[+{timer()}] Cleared video dir")

                fn, res = scanner()
                print(f"[+{timer()}] Received file from scanner")
                if not res:
                    done = True
                    return

                out_p = os.path.join(out_dir, os.path.splitext(os.path.basename(fn))[0])
                prog = 0.95
                images = gis.proc_image_persist(ws, wf.prompt, wf.param, wf.out_idx, fn, out_p)
                prog = 0.98
                if len(images) < 1:
                    done=True
                    return

                cr.update(fn, images, vid_path)
                prog = 1.0
                done = True
            except Exception as e:
                print(f"Failed to retreive data: {e}")
                done = True

        expected_time = 60
        st = time.time()
        thread = threading.Thread(target=fun)
        thread.start()
        cprog = 0
        while not done:
            upd = int((time.time() - st) * amount / expected_time - cprog) * (random.random() * 1.5)
            upd = int(min(upd, prog*amount-cprog))
            progress.update(upd)
            cprog = cprog + upd
            time.sleep(0.03)

        thread.join()
        progress.update(100)
        
        return cr.get()


    css = """
    footer {visibility: hidden}
    h1 { text-align : center; display : block;}
    h2 { text-align : center; display : block;}
    #logo_left { width: 100px; height: 100px; margin : auto;}
    """

    theme = gr.themes.Soft()
    
    with gr.Blocks(theme=theme, css = css) as inf:
        # with gr.Row():
        #     gr.Image(value=dhs_logo, elem_id="logo_left")
        #     #gr.Image(value=nat_logo, elem_id="logo_lft")
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
                vid_view = gr.Video(label="Evolution", autoplay=True, every=10)
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
