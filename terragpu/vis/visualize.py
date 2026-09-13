"""Legacy color and raster-export helpers; no model dependencies."""
import random
import numpy as np
import rasterio as rio

def gen_cmap(nclasses=19, random_colors=True):
    """
    Args:
        nclasses: integer with number of classes
        random:   boolean value, if True, colors are generated randomly
    Returns:
        colormap object with nclasses randomly generated colors
    """
    import matplotlib.colors as pltc
    cc = []
    if random_colors:
        all_colors = [k for k, v in pltc.cnames.items()]
        for color in random.sample(range(len(all_colors)), nclasses):
            cc.append(all_colors[color])
        return pltc.ListedColormap(cc)
    else:
        cc = ['gray', 'forestgreen', 'fuchsia', 'lemonchiffon', 'indigo',
              'lightcyan', 'mediumturquoise', 'navy', 'orange', 'pink',
              'rebeccapurple', 'sandybrown', 'silver', 'slategray',
              'springgreen', 'steelblue', 'tomato', 'violet', 'yellow'
              ]
        return pltc.ListedColormap(cc[:nclasses])

def get_RIT18_classes():
    classes = {
            'Other Class/Image Border':      'black',
            'Road Markings':                 'yellow',
            'Tree':                          'darkgreen',
            'Building':                      'gray',
            'Vehicle (Car, Truck, or Bus)':  'pink',
            'Person':                        'tomato',
            'Lifeguard Chair':               'firebrick',
            'Picnic Table':                  'brown',
            'Black Wood Panel':              'burlywood',
            'White Wood Panel':              'white',
            'Orange Landing Pad':            'orange',
            'Water Buoy':                    'powderblue',
            'Rocks':                         'lightslategrey',
            'Other Vegetation':              'lightgreen',
            'Grass':                         'darkorange',
            'Sand':                          'khaki',
            'Water (Lake)':                  'darkblue',
            'Water (Pond)':                  'dodgerblue',
            'Asphalt (Parking Lot/Walkway)': 'fuchsia'
    }
    return classes

def get_Vietnam_classes(cloud=False):
    classes = {
        'tree':        'green',
        'water':       'darkblue',
        'build':       'red',
        'shadow':      'purple',
        'other':       'linen',
        'small tree':  'lightgreen'
    }
    if cloud:
        classes['cloud'] = 'black'
    return classes

def npy_to_png(seg='segments.npy', classes='None', outimg='img.png', ret=True):
    """
    Args:
        seg: numpy array with anotated predictions
        classes: dict object with class:color for each class
        outimg: string with filename for png image
    Returns:
        saves png image
    """
    from PIL import Image
    from webcolors import name_to_rgb
    c = list(classes.values())  # list of colors from classes
    im = Image.new('RGB', seg.shape[::-1])
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            im.putpixel((j, i), name_to_rgb(c[int(seg[i, j])]))
    im.save(outimg)
    if ret:
        return im

def npy_to_tif(raster_f='image.tif', segments='segment.npy',
               outtif='segment.tif', ndval=-9999
               ):
    """
    Args:
        raster_f:
        segments:
        outtif:
    Returns:
    """
    # get geospatial profile, will apply for output file
    with rio.open(raster_f) as src:
        meta = src.profile
        nodatavals = src.read_masks(1).astype('int16')
    print(meta)

    # load numpy array if file is given
    if type(segments) == str:
        segments = np.load(segments)
    segments = segments.astype('int16')
    print(segments.dtype)  # check datatype

    nodatavals[nodatavals == 0] = ndval
    segments[nodatavals == ndval] = nodatavals[nodatavals == ndval]

    out_meta = meta  # modify profile based on numpy array
    out_meta['count'] = 1  # output is single band
    out_meta['dtype'] = 'int16'  # data type is float64

    # write to a raster
    with rio.open(outtif, 'w', **out_meta) as dst:
        dst.write(segments, 1)
