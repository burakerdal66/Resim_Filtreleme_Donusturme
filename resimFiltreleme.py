import gradio as gr
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Functions for different filters and image processing

def upload_image(image):
    height, width, _ = image.shape
    size_info = f"Width: {width} pixels, Height: {height} pixels"
    image_matrix = np.array(image)
    image_matrix_str = np.array2string(image_matrix, separator=', ')
    return image, size_info, image_matrix_str

def show_histogram(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = np.zeros(256)
    for value in img.flatten():
        hist[value] += 1

    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.grid()
    plt.savefig('histogram.png')
    plt.close()
    return 'histogram.png'

def show_histogram_rgb(image):
    color = ('b', 'g', 'r')
    hist = {c: np.zeros(256) for c in color}

    for i, col in enumerate(color):
        for value in image[:, :, i].flatten():
            hist[col][value] += 1

    plt.figure()
    plt.title("RGB Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    for col in color:
        plt.plot(hist[col], color=col)
        plt.xlim([0, 256])
    plt.grid()
    plt.savefig('histogram_rgb.png')
    plt.close()
    return 'histogram_rgb.png'

def show_histogram_luminance(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0]
    hist = np.zeros(256)
    for value in y.flatten():
        hist[value] += 1

    plt.figure()
    plt.title("Luminance Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist, color='black')
    plt.xlim([0, 256])
    plt.grid()
    plt.savefig('histogram_luminance.png')
    plt.close()
    return 'histogram_luminance.png'

def grayscale_image(image):
    gray = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            gray[i, j] = 0.299 * image[i, j, 2] + 0.587 * image[i, j, 1] + 0.114 * image[i, j, 0]
    return np.repeat(gray[:, :, np.newaxis], 3, axis=2)


def histogram_equalization(image):
    img = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf = cdf.astype('uint8')

    equ = cdf[img]
    equ = np.repeat(equ[:, :, np.newaxis], 3, axis=2)
    return equ


def contrast_enhancement(image):
    lab = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i, j, 2], image[i, j, 1], image[i, j, 0]
            l = 0.299 * r + 0.587 * g + 0.114 * b
            u = -0.147 * r - 0.289 * g + 0.436 * b
            v = 0.615 * r - 0.515 * g - 0.1 * b
            lab[i, j, :] = [l, u, v]

    l = lab[:, :, 0]
    clahe = create_CLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    enhanced_lab = lab.copy()
    enhanced_lab[:, :, 0] = cl

    enhanced_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            l, u, v = enhanced_lab[i, j, :]
            r = l + 1.13983 * v
            g = l - 0.39465 * u - 0.58060 * v
            b = l + 2.03211 * u
            enhanced_image[i, j, :] = [b, g, r]
    return enhanced_image


def create_CLAHE(clipLimit=2.0, tileGridSize=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    output = np.zeros_like(image)
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            for k in range(3):
                output[i, j, k] = np.clip(np.sum(kernel * image[i-1:i+2, j-1:j+2, k]), 0, 255)
    return output


def edge_detection(image, threshold_low=100, threshold_high=200):
    gray = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    edges = np.zeros_like(gray)
    for i in range(1, gray.shape[0] - 1):
        for j in range(1, gray.shape[1] - 1):
            gx = (gray[i - 1, j + 1] + 2 * gray[i, j + 1] + gray[i + 1, j + 1]) - (
                        gray[i - 1, j - 1] + 2 * gray[i, j - 1] + gray[i + 1, j - 1])
            gy = (gray[i + 1, j - 1] + 2 * gray[i + 1, j] + gray[i + 1, j + 1]) - (
                        gray[i - 1, j - 1] + 2 * gray[i - 1, j] + gray[i - 1, j + 1])
            magnitude = np.sqrt(gx ** 2 + gy ** 2)
            if magnitude > threshold_high:
                edges[i, j] = 255
            elif magnitude < threshold_low:
                edges[i, j] = 0
            else:
                edges[i, j] = 127
    edges = np.repeat(edges[:, :, np.newaxis], 3, axis=2)
    return edges


def sobel_edge_detection(image):
    gray = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    sobel_x = np.zeros_like(gray, dtype=np.float64)
    sobel_y = np.zeros_like(gray, dtype=np.float64)
    for i in range(1, gray.shape[0] - 1):
        for j in range(1, gray.shape[1] - 1):
            sobel_x[i, j] = (gray[i - 1, j + 1] + 2 * gray[i, j + 1] + gray[i + 1, j + 1]) - (
                        gray[i - 1, j - 1] + 2 * gray[i, j - 1] + gray[i + 1, j - 1])
            sobel_y[i, j] = (gray[i + 1, j - 1] + 2 * gray[i + 1, j] + gray[i + 1, j + 1]) - (
                        gray[i - 1, j - 1] + 2 * gray[i - 1, j] + gray[i - 1, j + 1])
    sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel = (sobel / np.max(sobel) * 255).astype(np.uint8)
    sobel = np.repeat(sobel[:, :, np.newaxis], 3, axis=2)
    return sobel


def laplacian_edge_detection(image):
    gray = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    laplacian = np.zeros_like(gray, dtype=np.float64)
    for i in range(1, gray.shape[0] - 1):
        for j in range(1, gray.shape[1] - 1):
            laplacian[i, j] = (
                        4 * gray[i, j] - gray[i - 1, j] - gray[i + 1, j] - gray[i, j - 1] - gray[i, j + 1])
    laplacian = np.abs(laplacian)
    laplacian = (laplacian / np.max(laplacian) * 255).astype(np.uint8)
    laplacian = np.repeat(laplacian[:, :, np.newaxis], 3, axis=2)
    return laplacian

def median_filter(image, kernel_size=5):
    padded_image = np.pad(image, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2), (0, 0)), 'constant', constant_values=0)
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(3):
                filtered_image[i, j, k] = np.median(padded_image[i:i+kernel_size, j:j+kernel_size, k])
    return filtered_image


def gaussian_filter(image, kernel_size=(15, 15), sigma=1.0):
    def gaussian(x, y, sigma):
        return (1.0 / (2.0 * np.pi * sigma ** 2)) * np.exp(- (x ** 2 + y ** 2) / (2.0 * sigma ** 2))

    kernel = np.fromfunction(lambda x, y: gaussian(x - kernel_size[0] // 2, y - kernel_size[1] // 2, sigma), kernel_size)
    kernel /= np.sum(kernel)
    padded_image = np.pad(image, ((kernel_size[0]//2, kernel_size[0]//2), (kernel_size[1]//2, kernel_size[1]//2), (0, 0)), 'constant', constant_values=0)
    blurred_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(3):
                blurred_image[i, j, k] = np.sum(kernel * padded_image[i:i+kernel_size[0], j:j+kernel_size[1], k])
    return blurred_image



def dilation(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    padded_image = np.pad(image, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2), (0, 0)), 'constant', constant_values=0)
    dilated = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(3):
                dilated[i, j, k] = np.max(padded_image[i:i+kernel_size, j:j+kernel_size, k])
    return dilated


def erosion(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    padded_image = np.pad(image, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2), (0, 0)), 'constant', constant_values=0)
    eroded = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(3):
                eroded[i, j, k] = np.min(padded_image[i:i+kernel_size, j:j+kernel_size, k])
    return eroded


def opening(image, kernel_size=5):
    eroded = erosion(image, kernel_size)
    opened = dilation(eroded, kernel_size)
    return opened


def closing(image, kernel_size=5):
    dilated = dilation(image, kernel_size)
    closed = erosion(dilated, kernel_size)
    return closed


def red_filter(image):
    r = image.copy()
    r[:, :, 1] = 0
    r[:, :, 2] = 0
    return r

def green_filter(image):
    g = image.copy()
    g[:, :, 0] = 0
    g[:, :, 2] = 0
    return g

def blue_filter(image):
    b = image.copy()
    b[:, :, 0] = 0
    b[:, :, 1] = 0
    return b

def show_rgb_values(image, evt: gr.SelectData):
    x, y = int(evt.index[0]), int(evt.index[1])
    color = image[y, x]
    return color.tolist()

def update_selection(image, evt: gr.SelectData):
    x1, y1 = int(evt.index[0]), int(evt.index[1])
    x2, y2 = x1 + int(evt.size[0]), y1 + int(evt.size[1])
    return x1, y1, x2, y2

def apply_filter(image, x1, y1, x2, y2, filter_type, filter_strength=1.0):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    if x1 == x2 or y1 == y2:  # If no area is selected
        return image
    roi = image[y1:y2, x1:x2]
    if filter_type == "Grayscale":
        roi = grayscale_image(roi)
    elif filter_type == "Histogram Equalization":
        roi = histogram_equalization(roi)
    elif filter_type == "Sharpen":
        roi = sharpen_image(roi)
    elif filter_type == "Blur":
        roi = gaussian_filter(roi)
    elif filter_type == "Contrast Enhancement":
        roi = contrast_enhancement(roi)
    elif filter_type == "Edge Detection":
        roi = edge_detection(roi)
    elif filter_type == "Sobel Edge Detection":
        roi = sobel_edge_detection(roi)
    elif filter_type == "Laplacian Edge Detection":
        roi = laplacian_edge_detection(roi)
    elif filter_type == "Median Filter":
        roi = median_filter(roi)
    elif filter_type == "Gaussian Filter":
        roi = gaussian_filter(roi)
    elif filter_type == "Dilation":
        roi = dilation(roi)
    elif filter_type == "Erosion":
        roi = erosion(roi)
    elif filter_type == "Opening":
        roi = opening(roi)
    elif filter_type == "Closing":
        roi = closing(roi)
    elif filter_type == "Red Filter":
        roi = red_filter(roi)
    elif filter_type == "Green Filter":
        roi = green_filter(roi)
    elif filter_type == "Blue Filter":
        roi = blue_filter(roi)

    # Applying filter strength
    if filter_strength != 1.0:
        roi = cv2.addWeighted(roi, filter_strength, image[y1:y2, x1:x2], 1 - filter_strength, 0)

    result = image.copy()
    result[y1:y2, x1:x2] = roi
    return result

def region_growing(image, seed_point, threshold=10):
    seed_x, seed_y = map(int, seed_point.split(','))
    h, w = image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)  # Updated mask dimensions
    _, _, mask, _ = cv2.floodFill(image.copy(), mask, (seed_x, seed_y), 255, (threshold,) * 3, (threshold,) * 3, cv2.FLOODFILL_FIXED_RANGE)
    mask = mask[1:-1, 1:-1]  # Remove the extra border from the mask
    return cv2.bitwise_and(image, image, mask=mask)

def active_contour_segmentation(image, iterations=2500):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    return segmented_image

def select_point(image, evt: gr.SelectData):
    x, y = int(evt.index[0]), int(evt.index[1])
    return f"{x},{y}"

# Gradio Interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy")
            image_output = gr.Image()
            size_info = gr.Textbox(label="Image Size")
            image_matrix = gr.Textbox(label="Image Matrix", lines=10, max_lines=20)

            image_input.upload(upload_image, image_input, [image_output, size_info, image_matrix])
            image_input.select(show_rgb_values, [image_input], [image_matrix])

        with gr.Column():
            filter_type = gr.Dropdown(
                ["Grayscale", "Histogram Equalization", "Sharpen", "Blur", "Contrast Enhancement", "Edge Detection",
                 "Sobel Edge Detection", "Laplacian Edge Detection", "Median Filter", "Gaussian Filter", "Dilation",
                 "Erosion", "Opening", "Closing", "Red Filter", "Green Filter", "Blue Filter"],
                label="Filter Type")
            x1 = gr.Number(label="x1")
            y1 = gr.Number(label="y1")
            x2 = gr.Number(label="x2")
            y2 = gr.Number(label="y2")
            filter_strength = gr.Slider(0.0, 1.0, value=1.0, label="Filter Strength")
            apply_button = gr.Button("Apply Filter")
            apply_button.click(apply_filter, [image_input, x1, y1, x2, y2, filter_type, filter_strength], image_output)

        with gr.Column():
            segment_type_region = gr.Markdown("## Region Growing Segmentation")
            point = gr.Textbox(label="Seed Point (x,y)", interactive=False)
            threshold = gr.Slider(0, 100, value=10, label="Region Growing Threshold")
            segment_button_region = gr.Button("Apply Region Growing Segmentation")
            segment_button_region.click(region_growing, [image_input, point, threshold], image_output)

            segment_type_active = gr.Markdown("## Active Contour Segmentation")
            iterations = gr.Number(value=2500, label="Active Contour Iterations")
            segment_button_active = gr.Button("Apply Active Contour Segmentation")
            segment_button_active.click(active_contour_segmentation, [image_input, iterations], image_output)

            histogram_button = gr.Button("Show Grayscale Histogram")
            histogram_button.click(show_histogram, image_input, image_output)

            histogram_rgb_button = gr.Button("Show RGB Histogram")
            histogram_rgb_button.click(show_histogram_rgb, image_input, image_output)

            histogram_luminance_button = gr.Button("Show Luminance Histogram")
            histogram_luminance_button.click(show_histogram_luminance, image_input, image_output)

        image_input.select(select_point, [image_input], [point])

demo.launch()
