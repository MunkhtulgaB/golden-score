import pyredner
import torch

objects = pyredner.load_obj('grip.obj', return_objects=True)
camera = pyredner.automatic_camera_placement(objects, resolution=(512, 512))

scene = pyredner.Scene(camera=camera, objects=objects)
img = pyredner.render_albedo(scene).double()
print(img.shape)

import matplotlib.pyplot as plt
plt.imshow(torch.pow(img, 1.0/2.2).cpu())
plt.show()

