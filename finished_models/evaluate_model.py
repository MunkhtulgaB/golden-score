"""
This file is based on the redner tutorial:
https://github.com/BachiLi/redner/blob/master/tutorials/01_optimize_single_triangle.py

I agree to their MIT License and reproduce it below:
=====================================================================================

MIT License

Copyright (c) 2018 Tzu-Mao Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
=====================================================================================
"""


# The Python interface of redner is defined in the pyredner package
import pyredner
import torch
import matplotlib.pyplot as plt
import numpy as np
import math

# Optimize three vertices of a single triangle
# We first render a target image using redner,
# then perturb the three vertices and optimize to match the target.

target_img = pyredner.imread('source_images/osoto_fonseca-removebg.png')
target = torch.tensor(target_img).double()
img_dim = (target.shape[0], target.shape[1])

objects = pyredner.load_obj('osoto_fonseca_1.obj', return_objects=True)
objects = [objects[0]]

def translate(orig, translation):
	x, y, z = orig
	dx, dy, dz = translation
	return (x + dx, y + dy, z + dz)

def rotate(orig, pivot, angle, axis=0):
	try:
		orig = orig.tolist()
	except:
		pass

	axes = [0, 1, 2]
	axes.remove(axis)

	a, b = (orig[ax] for ax in axes)
	oa, ob = (pivot[ax] for ax in axes)

	a1 = math.cos(angle) * (a-oa) - math.sin(angle) * (b-ob) + ob
	b1 = math.sin(angle) * (a-oa) + math.cos(angle) * (b-ob) + ob

	invariant = orig[axis]
	to_return = [a1, b1]
	to_return.insert(axis, invariant)
	return to_return

# objects[0].light_id = -1
# objects[0].material_id = 0

# We need to tell redner first whether we are using GPU or not.
pyredner.set_use_gpu(torch.cuda.is_available())

# Now we want to setup a 3D scene, represented in PyTorch tensors,
# then feed it into redner to render an image.

# First, we set up a camera.
# redner assumes all the camera variables live in CPU memory,
# so you should allocate torch tensors in CPU

cam = pyredner.automatic_camera_placement(objects, resolution = img_dim)
cam.position = torch.tensor(rotate(cam.position, (0, 0, 0), math.pi/4.5, axis=1))
cam.position = torch.tensor(rotate(cam.position, (0, 0, 0), math.pi/4, axis=0))
cam.position = torch.tensor(translate(cam.position, -0.25 * cam.position))
cam.look_at = cam.look_at + torch.tensor([0., -0.05, -0.05])

# Next, we setup the materials for the scene.
# All materials in the scene are stored in a single Python list.
# The index of a material in the list is its material id.
# Our simple scene only has a single grey material with reflectance 0.5.
# If you are using GPU, make sure to copy the reflectance to GPU memory.
mat_grey = pyredner.Material(\
    diffuse_reflectance = \
        torch.tensor([0.5, 0.5, 0.5], device = pyredner.get_device()))
# The material list of the scene
materials = [mat_grey]

# Next, we setup the geometry for the scene.
# 3D objects in redner are called "Shape".
# All shapes in the scene are stored in a single Python list,
# the index of a shape in the list is its shape id.
# Right now, a shape is always a triangle mesh, which has a list of
# triangle vertices and a list of triangle indices.
# The vertices are a Nx3 torch double tensor,
# and the indices are a Mx3 torch integer tensor.
# Optionally, for each vertex you can specify its UV coordinate for texture mapping,
# and a normal for Phong interpolation.
# Each shape also needs to be assigned a material using material id,
# which is the index of the material in the material array.
# If you are using GPU, make sure to store all tensors of the shape in GPU memory.


shape_triangle = pyredner.Shape(\
    vertices = objects[0].vertices,
    indices = objects[0].indices,
    uvs = None,
    normals = None,
    material_id = 0)

# Merely having a single triangle is not enough for physically-based rendering.
# We need to have a light source. Here we setup the shape of a quad area light source,
shape_light = pyredner.Shape(\
    vertices = cam.position + torch.tensor([
    						cam.position.tolist(),
    						rotate(cam.position, (0, 0, 0), math.pi/9, axis=1),
    						rotate(cam.position, (0, 0, 0), math.pi/9, axis=0),
    						rotate(
    							rotate(cam.position, (0, 0, 0), math.pi/9, axis=0), 
    							(0, 0, 0), 
    							math.pi/9, 
    							axis=1
    						)], device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2],[1, 3, 2]],
        dtype = torch.int32, device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 0)

# The shape list of our scene contains two shapes:
shapes = [shape_triangle, shape_light]

# Now we assign some of the shapes in the scene as light sources.
# All area light sources in the scene are stored in a single Python list.
# Each area light is attached to a shape using shape id, additionally we need to
# assign the intensity of the light, which is a length 3 double tensor in CPU. 
light = pyredner.AreaLight(shape_id = 1, 
                           intensity = torch.tensor([400.0,400.0,400.0]),
                           two_sided=True,
                           directly_visible=False)

area_lights = [light]
# Finally we construct our scene using all the variables we setup previously.
scene = pyredner.Scene(cam, shapes, materials, area_lights)
# All PyTorch functions take a flat array of PyTorch tensors as input,
# therefore we need to serialize the scene into an array. The following
# function does this. We also specify how many Monte Carlo samples we want to 
# use per pixel and the number of bounces for indirect illumination here
# (one bounce means only direct illumination).

scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 20,
    max_bounces = 1)

# Now we render the scene as our target image.
# To render the scene, we use our custom PyTorch function in pyredner/render_pytorch.py
# First setup the alias of the render function
render = pyredner.RenderFunction.apply
# # Next we call the render function to render.
# # The first argument is the seed for RNG in the renderer.
# img = render(0, *scene_args)
img = target

# This generates a PyTorch tensor with size [width, height, 3]. 
# The output image is in the GPU memory if you are using GPU.
# Now we save the generated image to disk.
pyredner.imwrite(img.cpu(), 'evaluation/target.exr')
pyredner.imwrite(img.cpu(), 'evaluation/target.png')
# Now we read back the target image we just saved, and copy to GPU if necessary
target = pyredner.imread('evaluation/target.exr').double()
if pyredner.get_use_gpu():
    target = target.cuda()

# Next we want to produce the initial guess. We do this by perturb the scene.
shape_triangle.vertices = torch.tensor(\
    objects[0].vertices,
    device = pyredner.get_device(),
    requires_grad = True) # Set requires_grad to True since we want to optimize this



# We need to serialize the scene again to get the new arguments.
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 20,
    max_bounces = 1)
# Render the initial guess
img = render(1, *scene_args).double()
# Save the image
pyredner.imwrite(img.cpu(), 'evaluation/init.png')

# Compute the difference and save the images.
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'evaluation/init_diff.png')
print(diff.pow(2).sum().item())
