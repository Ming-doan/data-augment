import os
import json
import uuid
import shutil
import cv2 as cv
import numpy as np
from PIL import Image
from abc import abstractmethod
import matplotlib.pyplot as plt
from ._interface import Interface
from typing import Literal, Callable
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# Utils for user 👇


def create_image_placeholder(image: cv.Mat, mode: Literal['black', 'white', 'top-pixels', 'bottom-pixels', 'left-pixels', 'right-pixels'] = 'black'):
    h, w, _ = image.shape
    if mode == 'black':
        return np.zeros(image.shape, dtype=np.uint8)
    elif mode == 'white':
        return np.ones(image.shape, dtype=np.uint8) * 255
    elif mode == 'top-pixels':
        top_pixels_line = image[0, :, :].copy()
        return np.stack([top_pixels_line] * h, axis=0)
    elif mode == 'bottom-pixels':
        bottom_pixels_line = image[h - 1, :, :].copy()
        return np.stack([bottom_pixels_line] * h, axis=0)
    elif mode == 'left-pixels':
        left_pixels_line = image[:, 0, :].copy()
        return np.stack([left_pixels_line] * w, axis=1)
    elif mode == 'right-pixels':
        right_pixels_line = image[:, w - 1, :].copy()
        return np.stack([right_pixels_line] * w, axis=1)
    else:
        raise ValueError(f'Unknown mode: {mode}')


def bboxs_callback(bboxs: list[list[int]], callback: Callable[[int, int, int, int], list[int]]):
    new_bboxs = []
    for bbox in bboxs:
        x, y, w, h = bbox
        new_bboxs.append(callback(x, y, w, h))
    return new_bboxs


# Utils for developer 👇


def image_cutter(image: cv.Mat, bboxs: list[list[int, int, int, int]]) -> list[cv.Mat]:
    images = []
    for bbox in bboxs:
        x, y, w, h = bbox
        x, y, w, h = round(x), round(y), round(w), round(h)
        images.append(image[y:y+h, x:x+w])
    return images


def images_placer(images: list[cv.Mat], bboxs: list[list[int, int, int, int]], original_image: cv.Mat) -> cv.Mat:
    for i, image in enumerate(images):
        bbox = bboxs[i]
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        original_image[y:y+h, x:x+w] = image
    return original_image


def draw_on_ax(ax: plt.Axes, image: cv.Mat, bboxs: list[list[int, int, int, int]], cats: list[str], title: str = None):
    ax.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    for i, bbox in enumerate(bboxs):
        x, y, w, h = bbox
        x, y, w, h = round(x), round(y), round(w), round(h)
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False,
                     edgecolor='red', linewidth=1))
        ax.text(x, y, cats[i], fontdict={'color': 'red', 'size': 10})
    if title is not None:
        ax.set_title(title)


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = Image.frombytes(
        'RGB', canvas.get_width_height(), canvas.tostring_rgb())
    return img


def show_table(data: list[list[str]], cols: list[str], max_length: int = 20):
    # Calculate max padding
    max_padd = [len(col) for col in cols]
    for row in data:
        for i, col in enumerate(row):
            if len(str(col)) > max_length:
                max_padd[i] = max_length
            elif len(str(col)) > max_padd[i]:
                max_padd[i] = len(str(col))
    # Trim data
    for i, row in enumerate(data):
        for j, col in enumerate(row):
            if len(str(col)) > max_length:
                data[i][j] = str(col)[:max_length - 3] + '...'
    # Show top border
    print('┌', end='')
    for i, col in enumerate(cols):
        print('─' * (max_padd[i] + 2), end='')
        if i != len(cols) - 1:
            print('┬', end='')
    print('┐')
    # Show columns
    print('│', end='')
    for i, col in enumerate(cols):
        print(f' {col}{" " * (max_padd[i] - len(str(col)))} │', end='')
    print()
    # Show divider
    print('├', end='')
    for i, col in enumerate(cols):
        print('─' * (max_padd[i] + 2), end='')
        if i != len(cols) - 1:
            print('┼', end='')
    print('┤')
    # Show content
    for row in data:
        print('│', end='')
        for i, col in enumerate(row):
            print(f' {col}{" " * (max_padd[i] - len(str(col)))} │', end='')
        print()
    # Show bottom border
    print('└', end='')
    for i, col in enumerate(cols):
        print('─' * (max_padd[i] + 2), end='')
        if i != len(cols) - 1:
            print('┴', end='')
    print('┘')


def logger(message: str, level: Literal['success', 'info', 'warning', 'error'] = 'info'):
    def _green(ssk):
        return f'\033[92m{ssk}\033[0m'

    def _blue(ssk):
        return f'\033[94m{ssk}\033[0m'

    def _yellow(ssk):
        return f'\033[93m{ssk}\033[0m'

    def _red(ssk):
        return f'\033[91m{ssk}\033[0m'

    if level == 'success':
        print(_green('[SUCCESS]'), end=' ')
        print(message)
    elif level == 'info':
        print(_blue('[INFO]'), end=' ')
        print(message)
    elif level == 'warning':
        print(_yellow('[WARNING]'), end=' ')
        print(message)
    elif level == 'error':
        print(_red('[ERROR]'), end=' ')
        print(message)
    else:
        raise ValueError('Invalid level')


def signature_checker(signature: str):
    if not all(ord(char) < 128 for char in signature):
        logger('Signature must be ASCII', level='error')
        raise ValueError('Signature must be ASCII')
    if len(signature) > 20:
        logger('Signature is too long', level='error')
        raise ValueError('Signature is too long')
    if not signature.isalnum():
        logger('Signature must be alphanumeric', level='error')
        raise ValueError('Signature must be alphanumeric')


class Default:
    Amount = 100
    Seed = 42
    ImageExtension = 'jpg'
    TextExtension = 'txt'


np.random.seed(Default.Seed)


# Abstract class
class Method:
    def __init__(self, *args, **kwargs):
        self.frame_cutter = False
        self.amount = Default.Amount if kwargs.get(
            'amount') is None else kwargs.get('amount')
        self.frac = kwargs.get('frac') if kwargs.get(
            'frac') is not None else None

    def set_amount(self, amount: int):
        self.amount = amount

    @abstractmethod
    def transform(self, images: Interface.Images,
                  bboxs: Interface.Bboxs, width: Interface.W, height: Interface.H) -> tuple[Interface.Images, Interface.Bboxs]: ...


class COCO:
    def __init__(self, directory: str, json_file: str):
        self.directory = directory
        self.json_file = json_file

        logger(f'Loading {json_file}...')
        try:
            with open(json_file, 'r') as f:
                self.data: dict[str, list[dict]] = json.load(f)
            logger(f'Loaded {json_file}', level='success')
        except Exception as e:
            logger(f'Failed to load {json_file}', level='error')
            raise e

        self.categories = self.data['categories']
        self.images = self.data['images']

    def __find_category(self, category_id: int) -> dict:
        for category in self.categories:
            if category['id'] == category_id:
                return category
        return None

    def __read_as_format(self, image: dict, annotations: list[dict], get_id: bool) -> Interface.ReadFormat:
        # Read image
        image_path = f'{self.directory}/{image["file_name"]}'
        _image = cv.imread(image_path)
        # Read width and height
        width, height = image['width'], image['height']
        # Read bboxs and categories
        bboxs = []
        categories = []
        for annotation in annotations:
            # Read bboxs
            bboxs.append([*annotation['bbox']])
            # Read categories
            category = self.__find_category(annotation['category_id'])
            if get_id:
                categories.append(category['id'])
            else:
                categories.append(category['name'])
        return _image, width, height, bboxs, categories

    def __get_by_id(self, image_id: int, get_id=False):
        image = self.data['images'][image_id]
        # Find annotations
        annotations = []
        for annotation in self.data['annotations']:
            if annotation['image_id'] == image['id']:
                annotations.append(annotation)
        return self.__read_as_format(image, annotations, get_id=get_id)

    def random(self):
        img_id = np.random.randint(0, len(self.images) - 1)
        return self.__get_by_id(img_id)

    def flow(self, size: int = Default.Amount, seed: int = Default.Seed):
        idx = np.random.permutation(len(self.images))[:size]
        for i in idx:
            yield self.__get_by_id(i, get_id=True)


class BaseWriter:
    def __init__(self, directory: str, **kwargs):
        self.directory = directory
        self.overwrite = kwargs.get('overwrite') if kwargs.get(
            'overwrite') is not None else True

        # Make directory
        self.make_dir(directory)

    def make_dir(self, directory: str):
        if not os.path.exists(directory):
            try:
                os.mkdir(directory)
            except Exception as e:
                raise e

    def clear_dir(self, directory: str):
        scan_obj = os.scandir(directory)
        for entry in scan_obj:
            if entry.is_dir():
                shutil.rmtree(entry.path)
            else:
                os.remove(entry.path)
        scan_obj.close()

    @abstractmethod
    def write(self, image: Interface.Images, width: Interface.W, height: Interface.H,
              bboxs: Interface.Bboxs, cats: list[int]) -> str:
        """
        Implement write image into folder.
        """


class YOLO(BaseWriter):
    def __init__(self, directory: str, **kwargs):
        super().__init__(directory, **kwargs)
        self.format = kwargs.get('format') if kwargs.get(
            'format') is not None else False

        # Make images directory
        self.make_dir(f'{self.directory}/images')

        # Make labels directory
        self.make_dir(f'{self.directory}/labels')

        # Warning for overwrite mode
        if self.overwrite:
            logger(
                f'Images and labels will be overwritten in {self.directory}', level='warning')
            # Clear images directory
            self.clear_dir(f'{self.directory}/images')
            # Clear labels directory
            self.clear_dir(f'{self.directory}/labels')

        # Warning for format mode
        if self.format:
            logger(
                f'Labels will be formatted in YOLO syntax', level='warning')

    def __format_yolo(self, bbox: list[int], width: Interface.W, height: Interface.H) -> list[int]:
        x, y, w, h = bbox
        x, y, w, h = x / width, y / height, w / width, h / height
        return x, y, w, h

    def write(self, image, width, height, bboxs, cats):
        # Create filename
        filename = uuid.uuid4().hex
        # Write image
        cv.imwrite(
            f'{self.directory}/images/{filename}.{Default.ImageExtension}', image)
        # Write labels
        with open(f'{self.directory}/labels/{filename}.{Default.TextExtension}', 'w') as f:
            for i, bbox in enumerate(bboxs):
                if self.format:
                    x, y, w, h = self.__format_yolo(bbox, width, height)
                else:
                    x, y, w, h = bbox
                    x, y, w, h = round(x), round(y), round(w), round(h)
                f.write(f'{cats[i]} {x} {y} {w} {h}\n')
        return filename


class CNN(BaseWriter):
    def __init__(self, directory: str, shape: tuple[int] = 112, **kwargs):
        super().__init__(directory, **kwargs)
        self.format = kwargs.get('format', False)
        self.shape = (shape, shape)
        self.mode = kwargs.get('mode', 'scale')  # 'keep', 'scale'
        self.threshold = kwargs.get('threshold', 10)

        # Make directory
        self.make_dir(directory)

    def write(self, image, width, height, bboxs, cats):
        # Split image
        images = image_cutter(image, bboxs)
        # Iterate over images
        for i, _image in enumerate(images):
            filename = uuid.uuid4().hex
            # Check if image not empty
            if _image.shape[0] < self.threshold or _image.shape[1] < self.threshold:
                continue
            cat = cats[i]
            # Create folder by cat name
            self.make_dir(f'{self.directory}/{str(cat)}')
            # Preprocessing image
            if self.mode == 'scale':
                _image = cv.resize(_image, self.shape)
            # Write image
            try:
                cv.imwrite(
                    f'{self.directory}/{str(cat)}/{filename}.{Default.ImageExtension}', _image)
            except Exception as e:
                logger(f'Failed to write image {filename}: {e}', level='error')
        return filename


# Distribution support
class DistRand:
    def __init__(self, range: tuple[int], mode: Literal['unif', 'norm', 'expo'] = 'unif', seed=Default.Seed):
        self.range = self.__get_random_range(range)
        self.mode = mode
        self.seed = seed

        # Get random function
        _rand_fn = self.__get_random_func()
        # Get random size
        _size = len(self.range)
        # Get random probabilities
        _probs = _rand_fn(size=_size)
        # Turn into positive
        _probs = np.abs(_probs)
        # Get histogram
        _hist, _ = np.histogram(_probs, bins=_size)
        # Normalize histogram
        _hist = _hist / _hist.sum()
        self.probs = _hist

        # Set random seed
        np.random.seed(self.seed)

    def __get_random_range(self, range: tuple[int]) -> np.ndarray:
        if len(range) == 2:
            if range[0] > range[1]:
                return np.arange(range[1], range[0], (range[0] - range[1]) / 10)[::-1]
            return np.arange(range[0], range[1], (range[1] - range[0]) / 10)
        elif len(range) == 3:
            if range[0] > range[1]:
                return np.arange(range[1], range[0], range[2])[::-1]
            return np.arange(range[0], range[1], range[2])
        raise ValueError(
            f'Invalid range. Expected 2 or 3 elements, got {len(range)}')

    def __get_random_func(self):
        if self.mode == 'unif':
            return np.random.uniform
        elif self.mode == 'norm':
            return np.random.normal
        elif self.mode == 'expo':
            return np.random.exponential
        else:
            raise ValueError(f'Unknown mode: {self.mode}')

    def rand(self) -> float:
        # Get random index
        return np.random.choice(self.range, p=self.probs)
