#!/usr/bin/env python
# coding: utf-8

"""Test out image manipulation techniques"""

import os

# set up env var for imagemagick (wand)
os.environ['MAGICK_HOME'] = '/opt/homebrew'

import cv2
import numpy as np
import skimage.util

from wand.image import Image
from wand.drawing import Drawing
from wand.color import Color
#from photutils.datasets import make_noise_image


#
# Global states
#
opdict = {}


#
# Helper functions
#


def image_op(opname):
    """Function decorator to register image operations"""
    def _deco(fn):
        opdict[opname] = fn
        return fn
    return _deco


def blend(orig_img: np.ndarray, add_img: np.ndarray, intensity: float|str = 0.5, mask: np.ndarray|None = None) -> np.ndarray:
    """Blend the add-on image to the original image, optionally only at pixels
    where the mask is True
    """
    if intensity == "min":
        out = np.minimum(orig_img, add_img)
    elif intensity == "max":
        out = np.maximum(orig_img, add_img)
    elif isinstance(intensity, float):
        out = cv2.addWeighted(orig_img, 1-intensity, add_img, intensity, 0)
    else:
        raise NotImplemented("Unknown intensity (%s) %s" % (type(intensity), intensity))
    if mask is not None:
        out = np.where(mask[..., np.newaxis], out, orig_img )
    return out



#
# Image processing functions
#
# Todo:
#   - Checkout https://github.com/DocCreator/DocCreator and https://doc-creator.labri.fr/
#   - Paper too thin seeing the shadow of the flip side?
#


@image_op("Shadow/watermark")
def shadow_defect_curved(img):
    shadow_intensity = 0.5
    num_points = 10
    white_page = 255 * np.ones_like(img)
    height, width = white_page.shape[:2]
    points = np.random.randint(0, height, size=(num_points, 2))
    points = points.reshape((-1, 1, 2))
    cv2.fillPoly(white_page, [points], color=(150, 150, 150))

    # Apply blur to the shadow region
    kernel_size = (25, 25)  # Adjust the kernel size for desired blur effect
    blurred_shadow = cv2.blur(white_page, kernel_size)

    # Create shadow image with blurred effect
    #shadow_image = cv2.addWeighted(img, 1 - shadow_intensity, blurred_shadow, shadow_intensity, 0)
    shadow_image = np.minimum(blurred_shadow, img)

    return shadow_image



@image_op("Hair defects")
def add_hair_defect(img):
    num_hairs=4
    hair_length_range=(800, 1000)
    hair_thickness_range=(1, 3)
    hair_color=(0, 0, 0)
    height, width, _ = img.shape

    img = img.copy()
    for _ in range(num_hairs):
        hair_length = np.random.randint(hair_length_range[0], hair_length_range[1])
        hair_thickness = np.random.randint(hair_thickness_range[0], hair_thickness_range[1])
        hair_angle = np.random.randint(0, 180)
        hair_start_x = np.random.randint(0, width)
        hair_start_y = np.random.randint(0, height)

        hair_end_x = int(hair_start_x + hair_length * np.cos(np.deg2rad(hair_angle)))
        hair_end_y = int(hair_start_y + hair_length * np.sin(np.deg2rad(hair_angle)))

        cv2.line(img, (hair_start_x, hair_start_y), (hair_end_x, hair_end_y), hair_color, hair_thickness)

    return img

@image_op("Hair wriggle")
def hair_defect_curve(img):
    num_hairs = 4
    hair_length_range = (800, 1000)
    hair_thickness_range = (1, 3)
    hair_color = (0, 0, 0)
    height, width, _ = img.shape

    img = img.copy()
    for _ in range(num_hairs):
        hair_length = np.random.randint(hair_length_range[0], hair_length_range[1])
        hair_thickness = np.random.randint(hair_thickness_range[0], hair_thickness_range[1])
        hair_angle = np.random.randint(0, 180)
        hair_start_x = np.random.randint(0, width)
        hair_start_y = np.random.randint(0, height)

        hair_end_x = int(hair_start_x + hair_length * np.cos(np.deg2rad(hair_angle)))
        hair_end_y = int(hair_start_y + hair_length * np.sin(np.deg2rad(hair_angle)))

        num_points = hair_length // 10
        t = np.linspace(0, 1, num_points)
        x = np.linspace(hair_start_x, hair_end_x, num_points) + np.random.randint(-10, 10, num_points)
        y = np.linspace(hair_start_y, hair_end_y, num_points) + np.random.randint(-10, 10, num_points)

        points = np.column_stack((x, y)).astype(np.int32)

        for i in range(len(points) - 1):
            cv2.line(img, tuple(points[i]), tuple(points[i+1]), hair_color, hair_thickness)

    return img


@image_op("Dust")
def add_dust_image(img, density=1e-4):
    num_particles = int(img.shape[0] * img.shape[1] * density)
    white_page = 255*np.ones_like(img)
    height, width= white_page.shape[:2]
    x = np.random.randint(0, width - 1, num_particles)
    y = np.random.randint(0, height - 1, num_particles)
    for n in range(num_particles):
        radius = 5
        cv2.circle(white_page, (x[n], y[n]), radius, (80, 90, 100), -1)
    dust_image = np.minimum(img, white_page)

    return dust_image

if False:
    # cannot use FreeType this unless you compiled OpenCV yourself
    # https://github.com/opencv/opencv-python/issues/117
    @image_op("Scribble")
    def scribble(img):
        ft = cv2.freetype.createFreeType2()
        ft.loadFontData(fontFileName="Shopping Script Demo.ttf", id=0)
        img_height, img_width, _ = img.shape
        max_x = img_width - 1
        max_y = img_height - 1
        ft.putText(img, "fascinating", (int(max_x * 0.5), int(max_y * 0.2)), 10, (0, 0, 0), 20, cv2.LINE_AA)
        ft.putText(img, "__", (int(max_x * 0.3), int(max_y * 0.5)), 12, (0, 0, 0), 15, cv2.LINE_AA)
        ft.putText(img, "_ _", (int(max_x * 0.15), int(max_y * 0.7)), 12, (0, 0, 0), 15, cv2.LINE_AA)
        ft.putText(img, "NB!", (int(max_x * 0.8), int(max_y * 0.3)), 10, (0, 0, 0), 20, cv2.LINE_AA)
        return img

if False:
    # OpenCV built-in font only
    @image_op("Scribble")
    def scribble(img):
        img_height, img_width, _ = img.shape
        max_x = img_width - 1
        max_y = img_height - 1
        cv2.putText(img, "fascinating", (int(max_x * 0.5), int(max_y * 0.2)), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 0), 20, cv2.LINE_AA)
        cv2.putText(img, "__", (int(max_x * 0.3), int(max_y * 0.5)), cv2.FONT_HERSHEY_SIMPLEX, 12, (0, 0, 0), 15, cv2.LINE_AA)
        cv2.putText(img, "_ _", (int(max_x * 0.15), int(max_y * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 12, (0, 0, 0), 15, cv2.LINE_AA)
        cv2.putText(img, "NB!", (int(max_x * 0.8), int(max_y * 0.3)), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 0), 20, cv2.LINE_AA)
        return img

@image_op("Scribble")
def scribble(img):
    height, width = img.shape[:2]
    texts = ["fascinating", "__", "_ _", "NB!", "V", "v", "mmm", "mnnm", "mmmmm", "|", "Z", "O", "o", "X"]
    with Drawing() as ctx, Image.from_array(img.astype(np.uint8)) as image:
        #ctx.font_family = 'Times New Roman, Nimbus Roman No9'  # for fonts installed in the system
        ctx.font = 'Shopping Script Demo.ttf'  # expect to find this TTF file locally
        ctx.font_size = 64
        ctx.gravity = "north_west"
        for text in texts:
            gray = np.random.randint(int(255*0.05), int(255*0.35))
            color = '#%02x%02x%02x' % (gray,gray,gray)
            x = np.random.randint(int(width*0.1), int(width*0.9))
            y = np.random.randint(int(height*0.1), int(height*0.9))

            ctx.fill_color = Color(color)
            ctx.stroke_color = Color(color)
            image.annotate(text, ctx, x, y)
        return np.array(image)


@image_op("Shadow")
def add_shadow(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = round(min(img.shape[:2])*1e-3)
    blur = cv2.GaussianBlur(gray, (0, 0), k)
    adjusted = cv2.addWeighted(blur, 1.2, -30, 0, 0)
    adjusted_bgr = cv2.cvtColor(adjusted, cv2.COLOR_GRAY2BGR)
    shadowed_img = cv2.addWeighted(img, 0.6, adjusted_bgr, 0.4, 0)

    return shadowed_img


@image_op("Binarize")
def binarize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)



@image_op("Dilate")
def dilate(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    k = round(min(img.shape[:2])*6e-4)
    kernel = np.ones((k, k), np.uint8)
    img = cv2.dilate(thresholded_image, kernel, iterations=1)
    return img


@image_op("Erode")
def dilate(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    k = round(min(img.shape[:2])*6e-4)
    kernel = np.ones((k, k), np.uint8)
    img = cv2.erode(thresholded_image, kernel, iterations=1)
    return img


# Perform opening (erosion followed by dilation)
@image_op("Opening")
def opening(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    k = round(min(img.shape[:2])*6e-4)
    kernel = np.ones((k, k), np.uint8)
    img = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)
    return img

# Perform closing (dilation followed by erosion)
@image_op("Closing")
def closing(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    k = round(min(img.shape[:2])*6e-4)
    kernel = np.ones((k, k), np.uint8)
    img = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)
    return img


@image_op("Watermark")
def watermark(img):
    # custom font see: https://gist.github.com/nathzi1505/904ce98d09e5f5785eb98d99171ab214
    transp_grey = (80, 80, 80)
    img2 = cv2.putText(img.copy(), "Watermark", (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 5, transp_grey, thickness=10)
    img = blend(img, img2, 0.6)
    return img


@image_op("Weak ink")
def weaken(img):
    # doesn't look good if low resolution,
    # and the radius/sigma param doesn't seem to make any effect
    with Image.from_array(img.astype(np.uint8)) as image:
        image.oil_paint()
        return np.array(image)


@image_op("Camera blur")
def camera_blur(img):
    # determine kernel size, must be positive odd integer
    k = 2*int(max(min(img.shape[:2])/200, 3))-1
    img = cv2.GaussianBlur(img, (k,k), 0)
    return img


@image_op("Add speckle noise")
def add_speckle_noise(img):
    """Add speckle noise to image, which is (img + R * img) for a Gaussian noise R"""
    noisy = skimage.util.random_noise(img, mode="speckle")
    noisy = (noisy * 255).astype(np.uint8)
    return noisy


@image_op("Add salt & pepper noise")
def add_sp_noise(img):
    noisy = skimage.util.random_noise(img, mode="s&p")
    noisy = (noisy * 255).astype(np.uint8)
    return noisy


@image_op("Add Gaussian noise")
def add_gaussian_noise(img):
    if "use skimage":
        noisy = skimage.util.random_noise(img, mode="gaussian")
        noisy = (noisy * 255).astype(np.uint8)
    if "use numpy" == False:
        mean, sigma = 0, np.sqrt(20)
        noise = np.random.normal(mean, sigma, img.shape)
        noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
    return noisy


@image_op("Add Poisson noise")
def add_poisson_noise(img):
    if "low-light" == False:
        # To simulate low-light noise - https://stackoverflow.com/questions/19289470/
        # alternative: https://stackoverflow.com/questions/22937589
        peak = 1.5  # positive, lower the darker the output
        noisy = np.random.poisson(img/255.0 * peak) / peak * 255
        return np.clip(noisy, 0, 255).astype(np.uint8)
    if "add layer" == False:
        peak = 1.5  # positive, lower the darker the output
        noisy = np.random.poisson(img/255.0 * peak) / peak * 255
        noisy = img + noisy
        return np.clip(noisy, 0, 255).astype(np.uint8)
    if "use skimage":
        noisy = skimage.util.random_noise(img, mode="poisson")
        noisy = (noisy * 255).astype(np.uint8)
    return noisy


@image_op("Rotate 5 deg clockwise")
def rotate_355(img):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    angle = -5  # Negative angle for clockwise rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_img


@image_op("Rotate 10 deg clockwise")
def rotate_350(img):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    angle = -10  # Negative angle for clockwise rotation
    # Perform rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_img


@image_op("Rotate 5 deg anti clockwise")
def rotate_5(img):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    angle = 5  # Negative angle for clockwise rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_img


@image_op("Rotate 10 deg anti clockwise")
def rotate_10(img):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    angle = 10  # Negative angle for clockwise rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_img


@image_op("Grayscale")
def grayscale(img):
    img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    return img


@image_op("Fax")
def fax(img):
    """Simulate FAX resolution. Standard resolution is 204x98 dpi, i.e.,
    vertical resolution was half of the horizontal. This function simulates a
    rectangular pixel that vertical is twice the horizontal size"""
    height, width = img.shape[:2]
    img = cv2.resize(img, (width, height//2), interpolation=cv2.INTER_NEAREST)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    return img


@image_op("Highlighter")
def add_highlight_defects(img):
    """The case of random highlighter mark on the paper"""
    num_defects = 50
    hilite_intensity = 0.9  # orig:hilite = 1:9
    height, width, _ = img.shape
    hilight_w = min(height, width)/10
    hilight_h = hilight_w*0.15
    hilite_color = np.array([0, 255, 0], dtype=np.uint8)  # highlighter color, should be high saturation
    hilite_mask = np.zeros((height, width), dtype=bool)

    for _ in range(num_defects):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        w = np.random.randint(int(0.6*hilight_w), int(hilight_w))
        h = np.random.randint(int(0.85*hilight_h), int(hilight_h))
        if np.mean(img[y:y+h, x:x+w]) < 255:
            hilite_mask[y:y+h, x:x+w] = True
    hilite_img = np.zeros_like(img)
    hilite_img[hilite_mask] = hilite_color

    img = blend(img, hilite_img, "min", mask=hilite_mask)
    return img
