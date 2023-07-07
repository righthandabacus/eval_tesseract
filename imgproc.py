#!/usr/bin/env python
# coding: utf-8

"""Test out image manipulation techniques"""

import os

# set up env var for imagemagick (wand)
os.environ['MAGICK_HOME'] = '/opt/homebrew'

import cv2
import numpy as np

from wand.image import Image
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
    dust_intensity = 0.3
    num_particles = int(img.shape[0] * img.shape[1] * density)
    white_page = 255*np.ones_like(img)
    height, width= white_page.shape[:2]
    x = np.random.randint(0, width - 1, num_particles)
    y = np.random.randint(0, height - 1, num_particles)
    for n in range(num_particles):
        radius = 5
        cv2.circle(white_page, (x[n], y[n]), radius, (80, 90, 100), -1)
    #dust_image = cv2.addWeighted(img, 1 - dust_intensity, white_page, dust_intensity, 0)
    dust_image = np.minimum(img, white_page)

    return dust_image

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

@image_op("Watermark")
def watermark(img):
    transp_grey = (80, 80, 80, 0.4)
    img = cv2.putText(img.copy(), "Watermark", (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 5, transp_grey, thickness=2)
    return img

@image_op("Salt and Pepper")
def snp(img):
    with Image.from_array(img.astype(np.uint8)) as image:
        # or Poisson noise type?
        image.noise("impulse", attenuate=0.5)
        return np.array(image)

@image_op("Weak ink")
def weaken(img):
    with Image.from_array(img.astype(np.uint8)) as image:
        image.oil_paint()
        return np.array(image)

@image_op("Shadow")
def add_shadow(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 5)
    adjusted = cv2.addWeighted(blur, 1.2, -30, 0, 0)
    adjusted_bgr = cv2.cvtColor(adjusted, cv2.COLOR_GRAY2BGR)
    shadowed_img = cv2.addWeighted(img, 0.6, adjusted_bgr, 0.4, 0)

    return shadowed_img

@image_op("Binarize")
def binarize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return img



@image_op("Dilate")
def dilate(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(thresholded_image, kernel, iterations=1)
    return img


@image_op("Erode")
def dilate(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(thresholded_image, kernel, iterations=1)
    return img


# Perform opening (erosion followed by dilation)
@image_op("Opening")
def opening(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)
    return img

# Perform closing (dilation followed by erosion)
@image_op("Closing")
def closing(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)
    return img

@image_op("Camera blur")
def camera_blur(img):
    img = cv2.GaussianBlur(img, (11,11), 0)
    # Update the image in-place
    img[:] = img
    return img


@image_op("Add Gaussian noise")
def add_noise(img):
    mean = 0
    std_dev = 30
    noise = np.random.normal(mean, std_dev, img.shape).astype(np.uint8)
    img += noise
    return img


@image_op("Add Poisson noise")
def add_poisson_noise(img):
    noise_intensity=0.5
    peak = 255.0
    normalized_img = img / 127.5 - 1.0
    lambda_param = np.clip(normalized_img * peak * noise_intensity, 0, None)
    lambda_param[np.isnan(lambda_param)] = 0
    poisson_noise = np.random.poisson(lambda_param) / peak * 255
    noisy_img = img + poisson_noise

    return noisy_img.astype((np.uint8))


#
# Image operations: All take an image, return a modified image with corresponding command
#


@image_op("Rotate 180 deg")
def fliplr(img):
    # alt.: img = cv2.rotate(img, cv2.ROTATE_180)
    img = cv2.flip(img, -1)
    return img, "img = cv2.flip(img, -1)"

@image_op("Rotate 5 deg clockwise")
def rotate_5_deg(img):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    angle = -5  # Negative angle for clockwise rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_img

@image_op("Rotate 10 deg clockwise")
def rotate_5_deg(img):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    angle = -10  # Negative angle for clockwise rotation
    # Perform rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_img

@image_op("Rotate 5 deg anti clockwise")
def rotate_5_deg(img):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    angle = 5  # Negative angle for clockwise rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_img

@image_op("Rotate 10 deg anti clockwise")
def rotate_5_deg(img):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    angle = 10  # Negative angle for clockwise rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_img

@image_op("Rotate 90 deg clockwise")
def fliplr(img):
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


@image_op("Rotate 90 deg counterclockwise")
def fliplr(img):
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


@image_op("Ink-stain")
def ink_stain(img):
    h, w, _ = img.shape

    # Generate random ink stains
    num_stains = np.random.randint(5, 10)  # Adjust the number of stains as needed
    for _ in range(num_stains):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2, y2 = np.random.randint(x1 + 10, w), np.random.randint(y1 + 10, h)
        color = np.random.randint(0, 256, 3).tolist()
        img=cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


@image_op("Grayscale")
def fliplr(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

@image_op("Reshape")
def reshape(img):
    img = cv2.resize(img, (int(img.shape[1] * 1.2), img.shape[0]))
    return img

@image_op("Highlighter")
def add_highlight_defects(img):
    """The case of random highlighter mark on the paper"""
    num_defects = 50
    defect_intensity = 0.1
    highlight_color = (0, 255, 0)  # highlighter color, should be high saturation
    height, width, _ = img.shape
    defect_image = 255*np.ones_like(img)
    highlight_color = np.array(highlight_color, dtype=np.uint8)

    for _ in range(num_defects):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        w = np.random.randint(300, 500)
        h = np.random.randint(60, 70)
        if np.mean(img[y:y+h, x:x+w]) < 255:
            defect_image[y:y+h, x:x+w] = highlight_color

    defective_image = cv2.addWeighted(img, 1 - defect_intensity, defect_image, defect_intensity, 0)
    return defective_image
