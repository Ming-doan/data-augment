from ._interface import Interface
from ._utils import *
# Transform methods
# Import your_method here 👇
from .brightness import Brightness
from .squeeze import Squeeze


class _BaseAugmentation:
    @abstractmethod
    def __init__(self, *args, **kwargs): ...

    def base_transform(self, method: Method, images: Interface.Images, bboxs: Interface.Bboxs, width: Interface.W, height: Interface.H):
        if method.frame_cutter:
            _images = image_cutter(images, bboxs)
            _images, new_bboxs = method.transform(
                _images, bboxs, width, height)
            images = images_placer(_images, bboxs, images)
        else:
            images, new_bboxs = method.transform(images, bboxs, width, height)
        return images, new_bboxs


class Augmentation(_BaseAugmentation):
    def __init__(self, methods: list[Method]):
        super().__init__()
        self.methods = self.__check_methods(methods)
        self.flow: COCO = None
        self.writer: YOLO = None

    def __check_methods(self, methods: list[Method]):
        if len(methods) == 0:
            logger('Methods is empty', level='error')
            raise ValueError('Methods is empty')
        for method in methods:
            if not issubclass(method.__class__, Method):
                logger(
                    f'{method.__class__.__name__} is not a subclass of Method', level='error')
                raise ValueError(
                    f'{method.__class__.__name__} is not a subclass of Method')
        return methods

    def __check_amount(self, amount: int):
        if amount <= 0:
            logger('Amount must be greater than 0', level='error')
            raise ValueError('Amount must be greater than 0')
        if amount > len(self.flow.images):
            logger(
                f'Amount must be less than or equal to {len(self.flow.images)}', level='error')
            raise ValueError(
                f'Amount must be less than or equal to {len(self.flow.images)}')

    def __set_fraction(self):
        for method in self.methods:
            if method.frac is not None:
                if method.frac > 1 or method.frac <= 0:
                    logger(
                        'Fraction must be greater than 0 and less than or equal to 1', level='error')
                    raise ValueError(
                        'Fraction must be greater than 0 and less than or equal to 1')
                amount = method.frac * len(self.flow.images)
                method.set_amount(int(amount))

    def __check_flow(self):
        if self.flow is None:
            logger('Flow is not set', level='error')
            raise ValueError('Flow is not set')

    def __set_writer(self, save_directory: str, **kwargs):
        if self.writer is not None:
            if self.writer.directory != save_directory:
                self.writer = YOLO(save_directory, **kwargs)
        else:
            self.writer = YOLO(save_directory, **kwargs)

    def __format(self, images: Interface.Images, bboxs: Interface.Bboxs):
        # Format images
        if isinstance(images, cv.Mat) or isinstance(images, np.ndarray):
            images = np.clip(images, 0, 255).astype(np.uint8)
        # Format bboxs
        fbboxs = []
        for bbox in bboxs:
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            fbboxs.append([x, y, w, h])
        return images, fbboxs

    def __display_progress(self, name: str, current: int, total: int, working_file: str, end='\r'):
        progress = int(current / total * 100)
        progress_bar = ''.join(['#' for _ in range(progress // 2)])
        remain_bar = ''.join(['-' for _ in range(50 - progress // 2)])
        print(
            f'{name}: {current}/{total} [{progress_bar}{remain_bar}] -> {working_file}', end=end)

    def flow_from_coco(self, directory: str, json_file: str):
        self.flow = COCO(directory, json_file)
        # Check methods amount
        for method in self.methods:
            self.__check_amount(method.amount)
        # Set fraction
        self.__set_fraction()

    def summary(self):
        # Show methods
        logger('Summary augmentation methods')
        _summary = [['---', '---', ''], ['Total', sum(
            [method.amount for method in self.methods]), '']]
        show_table([*[[method.__class__.__name__, method.amount, method.frame_cutter] for method in self.methods], *_summary], [
                   'Name', 'Amount', 'Frame cutter'])

    def preview(self):
        # Check if flow is set
        self.__check_flow()
        # Random select method
        method: Method = np.random.choice(self.methods)
        # Select random image
        img, w, h, bboxs, cats = self.flow.random()
        # Transform
        new_img, new_bboxs = self.base_transform(
            method, img.copy(), bboxs, w, h)
        # Format
        new_img, new_bboxs = self.__format(new_img, new_bboxs)
        # Init figure
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # Show original image
        draw_on_ax(ax[0], img, bboxs, cats)
        # Show transformed image
        draw_on_ax(ax[1], new_img, new_bboxs, cats)
        # Convert figure to PIL image
        pil_image = fig2img(fig)
        # Close figure
        plt.close(fig)
        # Return PIL image
        return pil_image

    def apply(self, save_directory: str, **kwargs):
        # Check if flow is set
        self.__check_flow()
        # Set writer
        self.__set_writer(save_directory, **kwargs)
        # Iterate over methods
        for method in self.methods:
            # Print progress
            self.__display_progress(
                method.__class__.__name__, 0, method.amount, 'Begin')
            progress_count = 0
            # Iterate over flow
            for imgs, w, h, bboxs, cats in self.flow.flow(size=method.amount):
                # Transform
                imgs, new_bboxs = self.base_transform(
                    method, imgs, bboxs, w, h)
                # Write to file
                filename = self.writer.write(imgs, w, h, new_bboxs, cats)
                # Print progress
                progress_count += 1
                self.__display_progress(
                    method.__class__.__name__, progress_count, method.amount, filename)
            # Reset progress
            self.__display_progress(
                method.__class__.__name__, progress_count, method.amount, f'Done{" "*28}', end='\n')
        # Final
        logger(
            f'Generate augmentation data successfully. Output at {save_directory}', level='success')


class Pipe(Method, _BaseAugmentation):
    def __init__(self, methods: list[Method], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.methods = self.__check_methods(methods)

    def __check_methods(self, methods: list[Method]):
        if len(methods) == 0:
            logger('Methods is empty', level='error')
            raise ValueError('Methods is empty')
        for method in methods:
            if not issubclass(method.__class__, Method):
                logger(
                    f'{method.__class__.__name__} is not a subclass of Method', level='error')
                raise ValueError(
                    f'{method.__class__.__name__} is not a subclass of Method')
        return methods

    def transform(self, images, bboxs, width, height):
        for method in self.methods:
            images, bboxs = self.base_transform(
                method, images, bboxs, width, height)
        return images, bboxs


class Original(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform(self, images, bboxs, width, height):
        return images, bboxs
