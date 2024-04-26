
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
