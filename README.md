
# VirtualMakeUp tool

This tool uses a variety of computer vision techniques and libraries to automatically detect the iris, the lips and the teeth on the image. Therefore you can:
- Whitten the teeth ;
- Change the color of the iris ;
- Change the color of the lips ;
- *Ongoing - not available yet:* Change the color of the hairs.

This tool is provided with three input images used for the tests and development. Refers to the `main()` method to add others and change it.

Below are the mask computed to detect pixel-wisely the iris, the lips and the teeth on 3 different images:

Input | Mask
:---: | :---:
![](https://github.com/JujuDel/VirtualMakeUp/blob/master/data/images/girl-no-makeup.jpg) | ![](https://github.com/JujuDel/VirtualMakeUp/blob/master/data/images/results/masks/girl-no-makeup.png)
![](https://github.com/JujuDel/VirtualMakeUp/blob/master/data/images/face1.png) | ![](https://github.com/JujuDel/VirtualMakeUp/blob/master/data/images/results/masks/face1.png)
![](https://github.com/JujuDel/VirtualMakeUp/blob/master/data/images/face2.png) | ![](https://github.com/JujuDel/VirtualMakeUp/blob/master/data/images/results/masks/face2.png)

## Results

Before | After
:---: | :---:
![](https://github.com/JujuDel/VirtualMakeUp/blob/master/data/images/girl-no-makeup.jpg) | ![](https://github.com/JujuDel/VirtualMakeUp/blob/master/data/images/results/whole/girl-no-makeup.png)
![](https://github.com/JujuDel/VirtualMakeUp/blob/master/data/images/face1.png) | ![](https://github.com/JujuDel/VirtualMakeUp/blob/master/data/images/results/whole/face1.png)
![](https://github.com/JujuDel/VirtualMakeUp/blob/master/data/images/face2.png) | ![](https://github.com/JujuDel/VirtualMakeUp/blob/master/data/images/results/whole/face2.png)

## How-to use the tool

### Dependencies

```python
import cv2
import copy
import dlib

import numpy as np

from tkinter import Tk, Button, Label
from tkinter import colorchooser
from PIL import Image, ImageTk
```
Make sure to install everything required.

To launch the tool:
```bash
> python virtualMakeup.py
```

### Some exemples:

#### Mouth

Lips | Lips & Teeth
:---: | :---:
![](https://github.com/JujuDel/VirtualMakeUp/blob/master/data/gifs/girl-no-makeup_Lips.gif) | ![](https://github.com/JujuDel/VirtualMakeUp/blob/master/data/gifs/face2_LipsTeeth.gif)

#### Eyes

![](https://github.com/JujuDel/VirtualMakeUp/blob/master/data/gifs/face2_Eyes.gif)
