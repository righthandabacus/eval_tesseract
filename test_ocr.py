"""OCR demo tool"""

import pytesseract
import pandas as pd


def ocr_details(img):
    """Run an image over PyTesseract

    Args:
        img: numpy array of image in RGB format of (H,W,C)

    Returns:
        A dict with the following keys, each is a list, all are integers except `text`:
        ['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num',
        'left', 'top', 'width', 'height', 'conf', 'text']
    """
    # oem 1 for LSTM model, 0 for legacy
    # psm 3 for fully automatic page segmentation
    data = pytesseract.image_to_data(img, lang="eng",
                                     config="--oem 1 --psm 3",
                                     output_type=pytesseract.Output.DICT)
    return data


if __name__ == "__main__":
    import argparse
    import json
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("input_img")
    args = parser.parse_args()

    imagefile = args.input_img
    img = cv2.cvtColor(cv2.imread(imagefile, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    data = ocr_details(img)
    #print(json.dumps(data, indent=4, ensure_ascii=False))
    df = pd.DataFrame(data)
    df = df[df["text"] != ""]
    with pd.option_context("display.max_rows", 100, "display.max_columns", 0, "display.min_rows", 30):
        print(df)
    strs = pytesseract.image_to_string(img, lang="eng",
                                       config="--oem 1 --psm 3")
    print(strs)
