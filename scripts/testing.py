"""
Test process of converting classification from raster to png.
Author: Jordan A. Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
"""
import sys
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from webcolors import name_to_rgb


def npy_to_png(img_array='segments.npy', classes='None',
               outimg='segments.png'
               ):

    water = (0, 0, 136)
    trees = (0, 60, 0)
    build = (255, 215, 0)
    shadow = (0, 255, 0)
    other = (222, 184, 135)
    smalltrees = (155, 0, 0)
    clouds = (0, 0, 0)

    colors = (trees, water, build, shadow, other, smalltrees, clouds)

    c = list(classes.values())
    print(c, colors)

    seg = img_array
    print("seg shape:", seg.shape[::-1])

    im = Image.new('RGB', seg.shape[::-1])
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            im.putpixel((j, i), name_to_rgb(c[int(seg[i, j])]))

    im.show(outimg)


np.set_printoptions(threshold=sys.maxsize)
data = np.load("subset1_8patches_64.npy", allow_pickle=True)

for i in range(data.shape[-1]):
    channel_min = data[:, :, i].min()
    channel_max = data[:, :, i].max()
    data[:, :, i] = 2.0 * (data[:, :, i] - channel_min) / \
        (channel_max - channel_min) - 1.0
    channel_min_post = data[:, :, i].min()
    channel_max_post = data[:, :, i].max()
    print(f'{i + 1},{channel_min},{channel_max}')

data1 = data[1]
print(data1.shape)

model = load_model("unet_vietnam_Adadelta_48000_64_0.0001_64_200-0.80.h5")

pred = model.predict(np.expand_dims(data1, 0))[0].argmax(axis=-1)
print(pred.shape)

classes = {
        'tree':        'darkgreen',
        'water':       'blue',
        'build':       'gray',
        'shadow':      'purple',
        'field/other': 'tan',
        'small tree':  'lightgreen',
        'cloud': 'black'
    }
npy_to_png(pred, classes)

# plt.imshow(pred)
# plt.show()
