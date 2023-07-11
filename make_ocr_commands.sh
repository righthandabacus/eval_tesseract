#!/bin/bash

for x in images-*/*.png; do
    DIR=`dirname "$x"`
    DIR=${DIR/images/json}
    BASENAME=`basename "$x" .png`
    mkdir -p "$DIR"
    echo "python run_ocr.py '$x' '$DIR/$BASENAME.json'"
done
