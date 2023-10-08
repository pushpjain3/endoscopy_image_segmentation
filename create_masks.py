import os
import cv2
import matplotlib.pyplot as plt
import json
import numpy as np

hues = {'bg': 0,'aw': 0,'lvr': 0,'git': 328,'fat': 58,'gspr': 80,'ct':19,'blood': 0,'cd': 60,'hook': 130,'gbldr': 357,'hv': 217,'llmt': 40}
        
saturation = {'bg': 100,'aw': 33.3,'lvr': 55.3,'git': 69.7,'fat': 59.7,'gspr':100,'ct':100,'blood': 100,'cd': 100,'hook': 33.7,'gbldr':37.3, 'hv': 100,'llmt': 100}
        
value = {'bg': 0,'aw': 82.4,'lvr': 100,'git': 90.6,'fat': 72.9,'gspr':100,'ct':100,'blood': 100,'cd': 100,'hook': 100,'gbldr': 100,'hv': 50.2,'llmt': 43.5}

classes = hues.keys()


def create_masks(img, shapes):
    
    blank = np.zeros(shape=im.shape, dtype=np.uint8)
    
    channels = []
    labels = [x['label'] for x in shapes]
    points = [np.array(x['points'], dtype=np.int32) for x in shapes]
    
    label_points = dict(zip(labels, points))

    for i, label in enumerate(classes):
        
        if label in labels:
            cv2.fillPoly(blank, [label_points[label]], (hues[label], saturation[label], value[label]))
            
    return cv2.cvtColor(blank, cv2.COLOR_HSV2RGB)

def get_shape(msk):
    
    with open(msk) as handle:
        data = json.load(handle)
    
    shapes = data['shapes']
    
    return shapes

images = sorted(os.listdir('images'), key=lambda x: int(x.split('.')[0]))
masks = sorted(os.listdir('masks'), key=lambda x: int(x.split('.')[0]))

for i, (img_fn, msk_fn) in enumerate(zip(images, masks)):
    
    img = cv2.imread(os.path.join('images', img_fn), 1)
    msk = os.path.join('annotated', msk_fn)
    shapes = get_shape(msk)
    img_color = create_masks(img, shapes)
    cv2.imwrite(img_fn+'_color_mask.png', img_color)