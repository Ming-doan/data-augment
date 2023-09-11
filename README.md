# Ultimate Data Generator

> This file is used to supporting images generation for data augmented. With optimize code flow for looping, trainsforming, visualizing and exporting, this project will strongly help for data augmentation.

## Clone this project ⚡

```bash
# Clone this project from github
git clone https://github.com/Ming-doan/data-augment.git drabridge
# Change directory
cd ./drabridge/
# Create virtualenv (optional, you can use your own environment)
pip install virtualenv
env/Scripts/activate
# Download requirement libraries
pip install -r requirements.txt
```

## Download data for augmented 🔥

👉Download from this [URL](https://github.com/makerviet/via-datasets/releases/download/v1.0/via-trafficsign-coco-20210321.zip)

After download, extract and move all contents into `coco/` folder. The folder structure look like:

```
coco/
    annotations/
    train/
    val/
logic/
yolo/
RUNME.ipynb
```

## Documentation 📖

### Before transforming

- Modify your `SIGNATURE`.
- The `SIGNATURE` must **NOT** contain space or special characters.

In `RUNME.ipynb` at cell `1`

```py
SIGNATURE = 'someFancyName'
```

### Configuation amount of sample on each methods

Define as constant amount. In `RUNME.ipynb` at cell `2`

```py
from logic import Original, Brightness
METHODS = [
    Original(amount=500),
    Brightness(amount=200)
]
```

Define as fraction of total data. In `RUNME.ipynb` at cell `2`

```py
from logic import Original, Brightness
METHODS = [
    Original(frac=0.7),
    Brightness(frac=0.5)
]
```

In the cell `3` of `RUNME.ipynb`. There are the table for check the total of sample. For example:

```
┌─────────────┬────────┬──────────────┐
│ Name        │ Amount │ Frame cutter │
├─────────────┼────────┼──────────────┤
│ Original    │ 7204   │ False        │
│ Brightness  │ 5146   │ False        │
│ ---         │ ---    │              │
│ Total       │ 12350  │              │
└─────────────┴────────┴──────────────┘
```

### Concat multiple methods

In `RUNME.ipynb` at cell `2`

```py
from logic import Original, Brightness, Squeeze, Pipe
METHODS = [
    Original(amount=500),
    Pipe([Brightness(), Squeeze()], amount=1000)
]
```

### Create new method

Create new file with your method in `logic/your_method.py`

```python
# In your_method.py
from ._utils import *

# Define your_method
class YourMethod(Method): # You must inherite Method class ...
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # ...and pass kwargs into it

    def transform(self, images, bboxs, width, height):

        # Your algorithms defined here ...


        return images, bboxs # You must return both of them
```

If your want to use `frame_cutter` mode, which cutting the image into smaller images by its bounding boxs.

```py
...
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.frame_cutter = True # Set this param
```

### Import method to use

In `logic/__init__.py`

```py
...
# Import your method into this file
from .your_method import YourMethod
```

In `RUNME.ipynb`, at cell `2`

```py
# Add your method into use
from logic import YourMethod
METHODS = [
    YourMethod(),
    ...
]
```

### Support features

Create random number by distribution

Distribution modes:

- `unif` (default): Uniform distribution
- `norm`: Normal distribution
- `expo`: Exponential distribution

```py
self.rand = DistRand(range=(start, stop, step), mode='norm')
```

Create a background image to fill black area

Filling modes:

- `black` (default): Fill bg as black solid color
- `white`: Fill bg as white solid color
- `top-pixels`: Fill bg as the top line of pixels
- `bottom-pixels`: Fill bg as the bottom line of pixels
- `left-pixels`: Fill bg as the left line of pixels
- `right-pixels`: Fill bg as the right line of pixels

```py
bg_img = create_image_placeholder(original_image, mode='right-pixels')
```
