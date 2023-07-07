Convert PDF to image

    pdftoppm -r 600 document.pdf prefix
    convert prefix-000001.ppm prefix-000001.png

Resample (change resolution):

    mogrify -density 600 -resample 200 image.png

Run OCR (PyTesseract)

    for x in images-200dpi/* ; do name=`basename $x .png`; python ../run_ocr.py $x json-200dpi/$name.json ; done &
