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

# Optimize three vertices of a single triangle
# We first render a target image using redner,
# then perturb the three vertices and optimize to match the target.

target_img = pyredner.imread('kouchi_photo_tori_small.jpg')
target = torch.tensor(target_img).double()
img_dim = (target.shape[0], target.shape[1])

objects = pyredner.load_obj('kouchi_tori.obj', return_objects=True)
objects = [objects[0]]



# objects[0].light_id = -1
# objects[0].material_id = 0

# We need to tell redner first whether we are using GPU or not.
pyredner.set_use_gpu(torch.cuda.is_available())

# Now we want to setup a 3D scene, represented in PyTorch tensors,
# then feed it into redner to render an image.

# First, we set up a camera.
# redner assumes all the camera variables live in CPU memory,
# so you should allocate torch tensors in CPU
cam = pyredner.Camera(position = torch.tensor([0.1, 0.20, 1.4]),
                      look_at = torch.tensor([0.1, 0.25, 0.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      fov = torch.tensor([45.0]), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = img_dim,
                      fisheye = False)

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
# similary to the previous triangle.
shape_light = pyredner.Shape(\
    vertices = torch.tensor([[-1.0, -1.0, 3.],
                             [ 1.0, -1.0, 3.],
                             [-1.0,  1.0, 3.],
                             [ 1.0,  1.0, 3.]], device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2],[1, 3, 2]],
        dtype = torch.int32, device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 0)

shape_light1 = pyredner.Shape(\
    vertices = torch.tensor([[-1.0, -1.0, 7.0],
                             [ 1.0, -1.0, 7.0],
                             [-1.0,  1.0, 7.0],
                             [ 1.0,  1.0, 7.0]], device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2],[1, 3, 2]],
        dtype = torch.int32, device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 0)

shape_light2 = pyredner.Shape(\
    vertices = torch.tensor([[-1.0, -1.0, 6.0],
                             [ 1.0, -1.0, 6.0],
                             [-1.0,  1.0, 6.0],
                             [ 1.0,  1.0, 6.0]], device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2],[1, 3, 2]],
        dtype = torch.int32, device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 0)
# The shape list of our scene contains two shapes:
shapes = [shape_triangle, shape_light, shape_light1, shape_light2]

# Now we assign some of the shapes in the scene as light sources.
# All area light sources in the scene are stored in a single Python list.
# Each area light is attached to a shape using shape id, additionally we need to
# assign the intensity of the light, which is a length 3 double tensor in CPU. 
light = pyredner.AreaLight(shape_id = 1, 
                           intensity = torch.tensor([5.0,5.0,5.0]),
                           two_sided=True,
                           directly_visible=False)
light1 = pyredner.AreaLight(shape_id = 2, 
                           intensity = torch.tensor([0.5,0.5,0.5]),
                           two_sided=True,
                           directly_visible=False)
light2 = pyredner.AreaLight(shape_id = 3, 
                           intensity = torch.tensor([0.5,0.5,0.5]),
                           two_sided=True,
                           directly_visible=False)

area_lights = [light, light1, light2]
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
pyredner.imwrite(img.cpu(), 'results/optimize_vertices_kouchi_tori_obj_exp0/target.exr')
pyredner.imwrite(img.cpu(), 'results/optimize_vertices_kouchi_tori_obj_exp0/target.png')
# Now we read back the target image we just saved, and copy to GPU if necessary
target = pyredner.imread('results/optimize_vertices_kouchi_tori_obj_exp0/target.exr').double()
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
pyredner.imwrite(img.cpu(), 'results/optimize_vertices_kouchi_tori_obj_exp0/init.png')

# Compute the difference and save the images.
diff = torch.abs(target - img)
print(target)
print(img)
print(diff)

pyredner.imwrite(diff.cpu(), 'results/optimize_vertices_kouchi_tori_obj_exp0/init_diff.png')
input('wait')

# Now we want to refine the initial guess using gradient-based optimization.
# We use PyTorch's optimizer to do this.
shape_triangle.vertices.requires_grad = False
to_opt = torch.split(shape_triangle.vertices, [10, 20, 10445])
print(to_opt)
to_opt[0].requires_grad = True
optimizer = torch.optim.Adam([to_opt[0]], lr=1e-3)

# Run 200 Adam iterations.
for t in range(50):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image
    scene_args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4, # We use less samples in the Adam loop.
        max_bounces = 1)
    # Important to use a different seed every iteration, otherwise the result
    # would be biased.
    img = render(t+1, *scene_args).double()
    # Save the intermediate render.
    pyredner.imwrite(img.cpu(), 'results/optimize_vertices_kouchi_tori_obj_exp0/iter_{}.png'.format(t))
    # Compute the loss function. Here it is L2.

    diff = (img - target).pow(2)
    max_diff = diff.max()
    normalized = (diff / max_diff)
    loss = normalized.sum()
    # print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients of the three vertices.
    # print('grad:', shape_triangle.vertices.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current three vertices.

# Render the final result.
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1)
img = render(202, *scene_args).double()

result_obj = shapes[0]
result_obj.vertices = result_obj.vertices.detach()
pyredner.save_obj(result_obj, f'results/optimize_vertices_kouchi_tori_obj_exp0/result.obj')

# Save the images and differences.
pyredner.imwrite(img.cpu(), 'results/optimize_vertices_kouchi_tori_obj_exp0/final.exr')
pyredner.imwrite(img.cpu(), 'results/optimize_vertices_kouchi_tori_obj_exp0/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/optimize_vertices_kouchi_tori_obj_exp0/final_diff.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/optimize_vertices_kouchi_tori_obj_exp0/iter_%d.png", "-vb", "20M",
    "results/optimize_vertices_kouchi_tori_obj_exp0/out.mp4"])
