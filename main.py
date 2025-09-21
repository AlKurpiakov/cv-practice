import argparse
import sys
import cv2 as cv
import numpy as np
import os

x1, y1, x2, y2 = -1, -1, -1, -1

def cli_argument_parser():
    """Парсер командной строки для получения параметров"""
    parser = argparse.ArgumentParser(description="Image processing tool")

    # Параметры командной строки
    parser.add_argument('-i', '--image',
                        help='Path to an image',
                        type=str,
                        required=True,
                        dest='image_path')

    parser.add_argument('-o', '--output',
                        help='Output file name',
                        type=str,
                        default='test_out.jpg',
                        dest='out_image_path')

    parser.add_argument('-m', '--mode',
                        help='Mode (res, sepia, vig, pixel, frame, texture)',
                        type=str,
                        default='image',
                        dest='mode')

    parser.add_argument('-c', '--coef',
                        help='Input coefficient for resolution change',
                        type=float,
                        dest='coef')

    parser.add_argument('-r', '--radius',
                        help='Input radius for vignette effect',
                        type=float,
                        dest='radius')

    parser.add_argument('-b', '--block',
                        help='Input block size for pixelation effect',
                        type=int,
                        dest='block')
    
    parser.add_argument('-f', '--frameWidth',
                        help='Input frame width',
                        type=int,
                        default=35,
                        dest='frameWidth')
    
    parser.add_argument('-t', '--texture',
                        help='Input texture type(aquarelle or blik)',
                        type=str,
                        default='blik',
                        dest='texture')

    return parser.parse_args()


def load_image(image_path):
    """Чтение изображения"""
    if image_path is None:
        raise ValueError('Empty path to the image')
    return cv.imread(image_path)

def show_image(text, image, image2):
    """Отображение оригинального и обработанного изображения"""
    if image is None:
        raise ValueError('Empty path to the image')
    scale_percent = 20  # 50% от оригинального размера
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized_output = cv.resize(image2, dim, interpolation=cv.INTER_AREA)

    resized_input = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    cv.imshow('Original Image', resized_input)
    cv.imshow(text, resized_output)
    cv.waitKey(0)
    cv.destroyAllWindows()

def sepia_filter(image):
    height, width = image.shape[:2]

    sepia_image = np.zeros((height, width, 3), np.uint8)

    R = image[:,:,2] 
    G = image[:,:,1] 
    B = image[:,:,0] 
    
    sepia_image[:,:,0] = np.clip(0.272 * R + 0.534 * G + 0.131 * B, 0, 255) 
    sepia_image[:,:,1] = np.clip(0.349 * R + 0.686 * G + 0.168 * B, 0, 255)
    sepia_image[:,:,2] = np.clip(0.393 * R + 0.769 * G + 0.189 * B, 0, 255)

    return sepia_image



def change_resolution(image, coef):
    old_height, old_width, _ = image.shape
    
    new_height, new_width = int(old_height*coef), int(coef*old_width)
    resized_img = np.zeros_like((new_height, new_width, 3), dtype=np.uint8)


    x_indexes = np.linspace(0, old_width-1, new_width).astype(int)
    y_indexes = np.linspace(0, old_height-1, new_height).astype(int)

    resized_img = image[np.ix_(y_indexes, x_indexes)]

    return resized_img

def vignette(image, sigma):
    rows, cols = image.shape[:2]
    X_resultant_kernel = cv.getGaussianKernel(cols, sigma)
    Y_resultant_kernel = cv.getGaussianKernel(rows, sigma)
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()

    processed_img = image.copy()

    for i in range(3):  # Apply to each channel
        processed_img[:,:,i] = processed_img[:,:,i] * mask

    return processed_img


def select_area(image):
    """Выбор области для пикселизации"""
    new_x, new_y, new_width, new_height = 0, 0, 0, 0

    def mouse_click(event, x, y, flags, param):
        nonlocal new_x, new_y, new_width, new_height
        if event == cv.EVENT_LBUTTONDOWN:
            new_x, new_y = x, y
        elif event == cv.EVENT_LBUTTONUP:
            new_width = x - new_x
            new_height = y - new_y

    cv.imshow('Area', image)
    cv.setMouseCallback('Area', mouse_click)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return (new_x, new_y, new_width, new_height)


def pixel(src_image, block_size, x, y, width, height):
    """Пикселизация изображения"""

    pixel_img = np.zeros_like(src_image)
    np.copyto(pixel_img, src_image)

    roi = pixel_img[y:y + height, x:x + width]
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = roi[i:i + block_size, j:j + block_size]
            color = np.mean(block, axis=(0, 1)).astype(np.uint8)
            roi[i:i + block_size, j:j + block_size] = color

    pixel_img[y:y + height, x:x + width] = roi
    return pixel_img




def add_frame(image, frame_width, color_num=[152,52,25]):
    new_pic = np.zeros_like(image)
    np.copyto(new_pic, image)
    
    rows, cols = image.shape[:2]

    new_pic[:frame_width, :] = color_num
    new_pic[rows-frame_width:, :] = color_num
    new_pic[frame_width:rows-frame_width, :frame_width] = color_num
    new_pic[frame_width:rows-frame_width, cols-frame_width:] = color_num
    
    return new_pic

    
def add_texture(image, tex_type):
    texture_path = f'textures/{tex_type}.jpg'
    
    if not os.path.exists(texture_path):
        print(f"Файл не существует: {texture_path}")
        return image
    
    texture = cv.imread(texture_path)
    
    if texture is None:
        print(f"Ошибка загрузки: {texture_path}")
        return image
    
    # Ресайз и приведение к RGB (если исходное изображение в RGB)
    texture = cv.resize(texture, (image.shape[1], image.shape[0]))
    texture = cv.cvtColor(texture, cv.COLOR_BGR2RGB)  # OpenCV загружает как BGR
    
    # Векторное сложение
    textured_image = (image * 0.7 + texture * 0.3).astype(np.uint8)
    
    return textured_image


def main():
    """Основная функция программы"""
    # Получаем аргументы из командной строки
    args = cli_argument_parser()

    # Загружаем изображение
    src_image = load_image(args.image_path)

    # Выбираем режим обработки изображения
    if args.mode == 'res':
        new_image = change_resolution(src_image, args.coef)
        text = 'Resolution image'
    elif args.mode == 'sepia':
        new_image = sepia_filter(src_image)
        text = 'Sepia image'
    elif args.mode == 'vig':
        new_image = vignette(src_image, args.radius)
        text = 'Vignette image'
    elif args.mode == 'pixel':
        x, y, width, height = select_area(src_image)
        new_image = pixel(src_image, args.block, x, y, width, height)
        text = 'Pixel image'
    elif args.mode == 'frame':
        new_image = add_frame(src_image, args.frameWidth)
        text = 'framed image'
    elif args.mode == 'texture':
        new_image = add_texture(src_image, args.texture)
        text = 'framed image'
    else:
        raise ValueError('Unsupported mode')

    show_image(text, src_image, new_image)
    
    cv.imwrite(args.out_image_path, new_image, [
        cv.IMWRITE_JPEG_QUALITY, 95,     
        cv.IMWRITE_JPEG_PROGRESSIVE, 1,  
        cv.IMWRITE_JPEG_OPTIMIZE, 1      
    ])


if __name__ == '__main__':
    sys.exit(main() or 0)