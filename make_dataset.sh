#!/bin/bash

# make variation of defects
for defect in "shadow" "hair" "dust" "scribble" "binarize" "dilate" "erode" "open" "close" "watermark" "weak ink" "blur" "speckle" "salt pepper" "gaussian" "poisson" "rotate -5" "rotate -10" "rotate 5" "rotate 10" "grayscale" "fax" "highlighter"; do
    for indir in "images-modern-600" "images-modern-200" "images-modern-150" "images-modern-100" "images-modern-75" "images-old-300"; do
        outdir="$indir-$defect"
        mkdir -p "$outdir"
        for img in "$indir"/*; do
            base=`basename "$img"`
            python imgproc.py "$defect" "$img" "$outdir/$base"
        done
    done
done

# run SRGAN
for defect in "shadow" "hair" "dust" "scribble" "binarize" "dilate" "erode" "open" "close" "watermark" "weak ink" "blur" "speckle" "salt pepper" "gaussian" "poisson" "rotate -5" "rotate -10" "rotate 5" "rotate 10" "grayscale" "fax" "highlighter"; do
    for indir in "images-modern-200" "images-modern-150" "images-modern-100" "images-modern-75" "images-old-300"; do
        origdir="$indir-$defect"
        outdir="$indir-$defect-realsrgan2x"
        python inference_realesrgan.py -n RealESRGAN_x2plus -i "$origdir" -o "$outdir"
    done
done
