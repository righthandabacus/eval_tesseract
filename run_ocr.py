# run_ocr.py
import argparse
import json

import cv2
import pytesseract


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


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("input_img")
    parser.add_argument("output_json")
    args = parser.parse_args(args=None)

    img = cv2.cvtColor(cv2.imread(args.input_img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    data = ocr_details(img)

    with open(args.output_json, "w") as fp:
        fp.write(json.dumps(data, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
