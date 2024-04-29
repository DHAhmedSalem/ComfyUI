import urllib.request


server_address = "127.0.0.1:8188"


def upload_image(server, impath):
    with open(impath, "rb") as fd:
        content = fd.read()

    body = {
        "image" : content
    }

    rep = urllib.request.Request()
