"""Gradio UI to display the output from PyTesseract"""
import json
import os
from typing import Iterable

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


__FILEDIR = os.path.dirname(os.path.realpath(__file__))
__IMAGEDIR = os.path.join(__FILEDIR, "images")
__JSONDIR = os.path.join(__FILEDIR, "json")
__LAST = {}
# image format that OpenCV can read
IMAGEEXT = [".jpg", ".jpeg", ".jpe", ".png", ".bmp", ".dlib", ".webp", ".pbm",
            ".pgm", ".ppm", ".pxm", ".pnm", ".tiff", ".tif",]


def listdir() -> Iterable[str]:
    """Read the image dir and yield out all image files"""
    for filename in os.listdir(__IMAGEDIR):
        if not os.path.isfile(os.path.join(__IMAGEDIR, filename)):
            continue
        if os.path.splitext(filename)[1].lower() not in IMAGEEXT:
            continue
        yield filename


def records2html(ocr_records):
    # transform into HTML
    html = ["<p>"]
    for n, item in enumerate(ocr_records):
        last_item = ocr_records[n-1] if n else None
        if last_item:
            # new paragraph if page, block, paragraph number all match
            if (last_item["page_num"], last_item["block_num"], last_item["par_num"]) != (item["page_num"], item["block_num"], item["par_num"]):
                html.append("</p><p>")
            # new line if line number match but same paragraph
            elif last_item["line_num"] != item["line_num"]:
                html.append("<br/>")
        if item["text"]:
            html.append(highlight_word(item["text"], item["conf"]))
    html.append("</p>")
    return " ".join(html)


def highlight_word(word, conf):
    # Determine the CSS style based on the confidence range
    if conf >= 95:
        styled_word = f"<span>{word}</span>"
    elif conf >= 80:
        styled_word = f"<span style='color: green;'>{word}</span>"
    elif conf >= 50:
        styled_word = f"<span style='color: orange;'>{word}</span>"
    else:
        styled_word = f"<span style='color: red;'>{word}</span>"
    return styled_word


def find_record(x, y, ocr_records):
    # loop over extracted word to find the first match that fits the clicked coordinate
    for rec in ocr_records:
        left, top, width, height = rec["left"], rec["top"], rec["width"], rec["height"]
        if left <= x <= left+width and top <= y <= top+height:
            return rec


def gr_image_click(event: gr.SelectData) -> dict:
    """Event handler on click on a loaded image that passed through the OCR"""
    x, y = event.index  # x and y coordinate of the click
    ocr_records = __LAST["record"]
    clicked_word = find_record(x, y, ocr_records)
    if clicked_word is None:
        return {}
    return clicked_word


def gr_select(filename: str):
    """On select of filename, show the image as well as the OCR result"""
    basename, _ = os.path.splitext(filename)
    imagepath = os.path.join(__IMAGEDIR, filename)
    jsonpath = os.path.join(__JSONDIR, basename+".json")
    assert os.path.isfile(imagepath), imagepath
    assert os.path.isfile(jsonpath), jsonpath
    # read the image as numpy array
    np_image = cv2.cvtColor(cv2.imread(imagepath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    # read the OCR data and massage it
    ocr_data = json.load(open(jsonpath))
    sort_order = "page_num block_num par_num line_num word_num".split()
    df_ocr = pd.DataFrame(ocr_data).sort_values(by=sort_order)
    df_ocr = df_ocr[df_ocr["text"] != ""]
    ocr_rec = df_ocr.to_dict("records")
    ocr_html = records2html(ocr_rec)
    # histogram on the confidence
    fig = plt.figure(figsize=(8,4))
    plt.hist(df_ocr["conf"], bins=20)
    plt.xlabel("Confidence")
    plt.ylabel("Word count")
    # keep state
    __LAST.update({
        "filename": filename,
        "record": ocr_rec
    })
    return np_image, ocr_html, fig


def gr_reload_filelist():
    """Event handler on click on the reload button"""
    return gr.Dropdown.update(choices=list(listdir()))


def gradio_create(projectname="Test out image operations"):
    """Create the Gradio app: Connects the components to action handlers"""
    with gr.Blocks(analytics_enabled=False, title=projectname, css=None) as demo:
        # inputs
        gr.HTML(r'<div><h1 style="position:relative;"><img src="static/synechron.png" width="100" height="200" style="float:right;" />%s</h1></div>' % projectname)
        # Create the input and output interfaces
        with gr.Row():
            filenames = gr.Dropdown(choices=list(listdir()), max_choices=1, label="Images from disk", scale=3)
            refresh = gr.Button("Reload", scale=1)
        with gr.Row():
            image = gr.Image(label="Uploaded Image", interactive=False)  # default type: numpy
            with gr.Column():
                ocr_text = gr.HTML(label="OCR Text")
                ocr_detail = gr.JSON(label="OCR Detail")
        ocr_histogram = gr.Plot(label= "Confidence histogram")

        # event handlers
        filenames.input(gr_select,
                        inputs=[filenames],
                        outputs=[image, ocr_text, ocr_histogram],
                        api_name="select_file")
        image.select(gr_image_click,
                     outputs=[ocr_detail])
        refresh.click(gr_reload_filelist,
                      inputs=[],
                      outputs=[filenames])
    return demo


# serve the app
if __name__ == "__main__":
    app = FastAPI()
    # for serving Synechron logo
    static_dir = os.path.join(__FILEDIR, "static")
    os.makedirs(static_dir, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    # let Gradio hook up itself to FastAPI
    app = gr.mount_gradio_app(app, gradio_create(), path="/")
    #gradio_create().launch() -- only if no FastAPI needed, e.g., no static file
    #uvicorn.run("gradio_ocr:app", host="0.0.0.0", port=7860, reload=True)  # allow reload
    uvicorn.run(app, host="0.0.0.0", port=7860)  # cannot do reload=True
