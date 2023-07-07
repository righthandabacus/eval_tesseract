"""Imagemagick will not set DPI attribute in images but it is present in many formats. This is to fill in the value"""
import argparse
import wand.image

parser = argparse.ArgumentParser()
parser.add_argument("input_img")
parser.add_argument("dpi", type=int, default=600)
args = parser.parse_args()

with wand.image.Image(filename=args.input_img) as img:
    img.resolution = (args.dpi, args.dpi)
    img.save(filename=args.input_img)
