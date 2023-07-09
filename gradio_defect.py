#!/usr/bin/env python
# coding: utf-8

"""Test out image manipulation techniques for synthetic defects"""

import os
import shutil
from typing import Iterable

# set up env var for imagemagick (wand)
os.environ['MAGICK_HOME'] = '/opt/homebrew'

import cv2
import gradio as gr
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# all image manipulation is created in this file
from imgproc import opdict


__FILEDIR = os.path.dirname(os.path.realpath(__file__))
__IMAGEDIR = os.path.join(__FILEDIR, "images")
# image format that OpenCV can read
IMAGEEXT = [".jpg", ".jpeg", ".jpe", ".png", ".bmp", ".dlib", ".webp", ".pbm",
            ".pgm", ".ppm", ".pxm", ".pnm", ".tiff", ".tif",]


#
# Global states
#
state = dict(filename=None, orig=None)


#
# Helper functions
#


def listdir() -> Iterable[str]:
    """Read the image dir and yield out all image files"""
    for filename in sorted(os.listdir(__IMAGEDIR)):
        if not os.path.isfile(os.path.join(__IMAGEDIR, filename)):
            continue
        if os.path.splitext(filename)[1].lower() not in IMAGEEXT:
            continue
        yield filename


def getfilename(filename):
    """Return a usable filename based on the tentative filename"""
    base, ext = os.path.splitext(filename)
    counter = None
    while True:
        if counter is None:
            testfilename = base + ext
        else:
            testfilename = f"{base}.{counter}{ext}"
        path = os.path.join(__IMAGEDIR, testfilename)
        if not os.path.isfile(path):
            return testfilename
        else:
            counter = 1 if counter is None else counter+1


def load_image(filename: str) -> None:
    """Read the image file from disk and reset states: Image always in RGB
    since it is how Gradio expects it
    """
    img = cv2.imread(os.path.join(__IMAGEDIR, filename), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    state.update({
        "filename": filename,
        "orig": img,
    })


#
# UI actions for Gradio
#


def gr_fileupload(files):
    """Move all uploaded file to local dir"""
    for file in files:
        localpath = file.name
        realfilename = getfilename(os.path.basename(localpath))
        targetpath = os.path.join(__IMAGEDIR, realfilename)
        shutil.move(localpath, targetpath)
    return gr.Dropdown.update(choices=list(listdir()))


def gr_select(filename: str):
    """On select of filename, reset the screen"""
    load_image(filename)
    return state["orig"], None


def gr_action(operation):
    """On action button, update the image"""
    if state["orig"] is None or operation not in opdict:
        return
    print("Applying", operation, "to", state["filename"])
    function = opdict[operation]
    assert state["orig"].dtype == np.uint8
    img = function(state["orig"])
    assert img.dtype == np.uint8
    return img



def gradio_create(projectname="Test out image operations"):
    """Create the Gradio app: Connects the components to action handlers"""
    with gr.Blocks(analytics_enabled=False, title=projectname, css=None) as demo:
        # inputs
        gr.HTML(r'<div><h1 style="position:relative;"><img src="static/synechron.png" width="100" height="200" style="float:right;" />%s</h1></div>' % projectname)
        with gr.Row():
            filenames = gr.Dropdown(choices=list(listdir()), max_choices=1, label="Images from disk", scale=3)
            ops = gr.Dropdown(choices=list(opdict.keys()), max_choices=1, label="Operations", scale=2)
            run = gr.Button("Action", scale=1)
        # output
        with gr.Row():
            img1 = gr.Image(label="Original", show_label=True)
            img2 = gr.Image(label="Modified", show_label=True)
        fileupload = gr.Files(file_types=IMAGEEXT,
                              label="Upload new images")
        # action hook
        fileupload.upload(gr_fileupload,
                          inputs=[fileupload],
                          outputs=[filenames],
                          api_name="upload_file")
        filenames.input(gr_select,
                        inputs=[filenames],
                        outputs=[img1, img2],
                        api_name="select_file")
        run.click(gr_action,
                  inputs=[ops],
                  outputs=[img2],
                  api_name="apply")

    return demo


app = FastAPI()
# for serving Synechron logo
static_dir = os.path.join(__FILEDIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")
# let Gradio hook up itself to FastAPI
app = gr.mount_gradio_app(app, gradio_create(), path="/")

# serve the app
if __name__ == "__main__":
    #gradio_create().launch() -- only if no FastAPI needed, e.g., no static file
    #uvicorn.run(app, host="0.0.0.0", port=7860)  # cannot do reload=True
    uvicorn.run("gradio_defect:app", host="0.0.0.0", port=7860, reload=True)  # allow reload


# ----
#  create actions without user interaction
#    1. gr.RegisterEvent(fn=function, event="my_event_name", inputs=[x], outputs=[y])
#    2. gr.TriggerEvent("my_event_name")

# http://localhost:7860
