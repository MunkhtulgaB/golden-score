# -*- coding: utf-8 -*-
"""fast_local_shading.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/BachiLi/redner/blob/master/tutorials/fast_local_shading.ipynb

This tutorial focuses on the fast local shading mode of redner. This mode does not compute shadow or global illumination, but is faster and less noisy.

Again, we will import pyredner and pytorch, download the teapot object, load it, and setup the camera.
"""

import pyredner
import torch
import matplotlib.pyplot as plt
import urllib
import zipfile

objects = pyredner.load_obj('grip.obj', return_objects=True)

camera = pyredner.automatic_camera_placement(objects, resolution=(480, 640))

scene = pyredner.Scene(camera = camera, objects = objects)

"""Now, in contrast to the previous tutorials, we also setup some lightings. In the fast local shading mode, redner supports four kinds of lights: ambient light, point light, directional light, and spot light. We setup a point light with squared distance falloff at between the camera and the teapot this time."""

light = pyredner.PointLight(position = (camera.position + torch.tensor((0.0, 0.0, 100.0))).to(pyredner.get_device()),
                                                intensity = torch.tensor((20000.0, 30000.0, 20000.0), device = pyredner.get_device()))

# Commented out IPython magic to ensure Python compatibility.
img = pyredner.render_deferred(scene = scene, lights = [light])
# Visualize img
from matplotlib.pyplot import imshow
# %matplotlib inline
# Gamma correction to convert the image from linear space to sRGB
imshow(torch.pow(img, 1.0/2.2).cpu())

"""You may notice that the teapot rendering has some weird stripe patterns. This is because the teapot has a low number of polygons, and the normals used to calculate the Lambertian response introduces the pattern to rendering. A remedy for this is to use [Phong normal interpolation](https://en.wikipedia.org/wiki/Phong_shading): we compute a normal field at each vertex on the triangle mesh, and interpolate from the nearby vertices when computing the shading normal. Many Wavefront object files come with vertex normals, but this teapot does not. Redner implements [Nelson Max's algorithm](https://escholarship.org/content/qt7657d8h3/qt7657d8h3.pdf?t=ptt283) for computing the vertex normal. We can attach vertex normals as follows:"""

for obj in objects:
    obj.normals = pyredner.compute_vertex_normal(obj.vertices, obj.indices)

"""Re-render the scene, we get a smoother teapot:"""

scene = pyredner.Scene(camera = camera, objects = objects)
img = pyredner.render_deferred(scene = scene, lights = [light])
imshow(torch.pow(img, 1.0/2.2).cpu())

"""Apart from point lights, redner also supports ambient light, which is just a multiplier over the albedo values."""

light = pyredner.AmbientLight(intensity = torch.tensor((0.8, 1.2, 0.8), device = pyredner.get_device()))
img = pyredner.render_deferred(scene = scene, lights = [light])
imshow(torch.pow(img, 1.0/2.2).cpu())

"""Directional lights doesn't exhibit distance falloff:"""

light = pyredner.DirectionalLight(direction = torch.tensor((1.0, -1.0, 1.0), device = pyredner.get_device()),
                                                         intensity = torch.tensor((2.0, 3.0, 2.0), device = pyredner.get_device()))
img = pyredner.render_deferred(scene = scene, lights = [light])
imshow(torch.pow(img, 1.0/2.2).cpu())

"""And spot light is a directional light with exponential falloff over a direction."""

light = pyredner.SpotLight(position = camera.position.to(pyredner.get_device()),
                                               spot_direction = torch.tensor((0.0, 0.0, 1.0), device = pyredner.get_device()),
                                               spot_exponent = torch.tensor(100.0, device = pyredner.get_device()),
                                               intensity = torch.tensor((2.0, 3.0, 2.0), device = pyredner.get_device()))
img = pyredner.render_deferred(scene = scene, lights = [light])
imshow(torch.pow(img, 1.0/2.2).cpu())

"""If you want to adapt the deferred shading functionality, such as using materials other than Lambertian, you can write the shading code yourself in PyTorch. redner provides the `render_g_buffer` function to output different channels for you:"""

img = pyredner.render_g_buffer(scene = scene, channels = [pyredner.channels.position,
                                                         pyredner.channels.shading_normal,
                                                         pyredner.channels.diffuse_reflectance])
pos = img[:, :, :3]
normal = img[:, :, 3:6]
albedo = img[:, :, 6:9]

plt.figure()
pos_vis = (pos - pos.min()) / (pos.max() - pos.min())
imshow(pos_vis.cpu())
plt.figure()
normal_vis = (normal - normal.min()) / (normal.max() - normal.min())
imshow(normal_vis.cpu())
plt.figure()
imshow(torch.pow(albedo, 1.0/2.2).cpu())
plt.show()
"""You can then write your own PyTorch functions to assemble different channels. Antialising can be achieved using full-screen anti-aliasing (FSAA) -- just render the image at higher resolution then downsample it. See the source code of `render_g_buffer` to see how I did anti-aliasing."""