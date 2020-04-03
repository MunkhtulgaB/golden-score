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
This file makes use of SMPLify-x: 

@inproceedings{SMPL-X:2019,
  title = {Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
  author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}

By License agreement, I do not copy their Model & Software and the license. See more on:
https://github.com/vchoutas/smplify-x/blob/master/LICENSE

"""

import pyredner
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
from human_body_prior.body_model.body_model_vposer import BodyModelWithPoser
from scipy.ndimage import gaussian_filter
from human_body_prior.tools.model_loader import load_vposer as poser_loader

NUM_ITERS = 50
MODEL_NAME = 'uke_osotogari_filled'

# Get the body model
expr_dir = '../SMPL-X/vposer_v1_0' #'TRAINED_MODEL_DIRECTORY'  in this directory the trained model along with the model code exist
bm_path =  '../SMPL-X/models/smplx/SMPLX_MALE.npz'#'PATH_TO_SMPLX_model.npz'  obtain from https://smpl-x.is.tue.mpg.de/downloads
mano_exp_dir = expr_dir

bm = BodyModelWithPoser(bm_path=bm_path, batch_size=1, poser_type='vposer', smpl_exp_dir=expr_dir)

# Start from an approximation loaded from pickled SMPLify-x output
try: 
	body_model_path = 'results/' + MODEL_NAME + '/poZ_body.pkl'
	print('Loading pose from:', body_model_path)
	body_pickle = open(body_model_path, 'rb')
	body = pickle.load(body_pickle)
	pose_body = body.get('body_pose')
except:
	print('No existing pose found starting from original')
	body_model_path = 'input/source/' + MODEL_NAME + '/'
	body_pickle = open(body_model_path + '000.pkl', 'rb')
	body = pickle.load(body_pickle)

	# hand left
	bm.poser_handL_pt, bm.poser_handL_ps = poser_loader(mano_exp_dir)
	bm.poser_handL_pt.to(bm.trans.device)

	poZ_handL = bm.pose_hand.new(body.get('left_hand_pose'))
	bm.register_parameter('poZ_handL', torch.nn.Parameter(poZ_handL, requires_grad=True))

    # hand right
	bm.poser_handR_pt, bm.poser_handR_ps = poser_loader(mano_exp_dir)
	bm.poser_handR_pt.to(bm.trans.device)

	poZ_handR = bm.pose_hand.new(body.get('right_hand_pose'))
	bm.register_parameter('poZ_handR', torch.nn.Parameter(poZ_handR, requires_grad=True))
	bm.pose_hand.requires_grad = False

	pose_body = torch.tensor(body.get('body_pose')[0])


random_modification = torch.empty(pose_body.shape).normal_(mean=1,std=0.01)
poZ_body = torch.tensor(pose_body * random_modification, requires_grad=True)
bm.register_parameter('poZ_body', torch.nn.Parameter(poZ_body, requires_grad=True))


# Prepare to render
target_img = pyredner.imread('input/targets/' + MODEL_NAME + '.png')
# target_img = gaussian_filter(target_img, sigma=10)
target = torch.tensor(target_img).double()
target *= 20
pyredner.imwrite(target.cpu(), 'results/' + MODEL_NAME + '/target.png')

img_dim = (target.shape[0], target.shape[1])

cam_pos = torch.tensor([2.0, -0.3, 1.3], requires_grad=True)
cam = pyredner.Camera(position = cam_pos,
                      look_at = torch.tensor([-0.3, -0.6, 0.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      fov = torch.tensor([45.0]), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = img_dim,
                      fisheye = False)

mat_grey = pyredner.Material(\
    specular_reflectance = \
        torch.tensor([2.8, 2.8, 2.8], device = pyredner.get_device()))
# The material list of the scene
materials = [mat_grey]


shape_light = pyredner.Shape(\
    vertices = torch.tensor([[3.0, -1.0, -1.],
                             [3.0, -1.0, 2.],
                             [3.0,  1.0, -1.],
                             [3.0,  1.0, 2.]], device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2],[1, 3, 2]],
        dtype = torch.int32, device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 0)

shape_light1 = pyredner.Shape(\
    vertices = torch.tensor([[-2.0, -1.0, -4.],
                             [ 0.0, -1.0, -4.],
                             [-2.0,  1.0, -4.],
                             [ 0.0,  1.0, -4.]], device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2],[1, 3, 2]],
        dtype = torch.int32, device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 0)


light = pyredner.AreaLight(shape_id = 1, 
                           intensity = torch.tensor([6.0,6.0,6.0]),
                           two_sided=True,
                           directly_visible=False)

light1 = pyredner.AreaLight(shape_id = 2, 
                           intensity = torch.tensor([4.0,4.0,4.0]),
                           two_sided=True,
                           directly_visible=False)

vertices = bm.forward().v[0]
indices = bm.f
shape_triangle = pyredner.Shape(
	vertices = vertices, 
	indices = bm.f, 
	material_id=0)


shapes = [shape_triangle, shape_light, shape_light1]
area_lights = [light, light1]
scene = pyredner.Scene(cam, shapes, materials, area_lights)

# Optimise
# optimizer = torch.optim.SGD([bm.poZ_body], lr=1, momentum=0.9)
optimizer = torch.optim.Adam([bm.poZ_body], lr=5e-2)
# optimizer = torch.optim.Adam([cam.position], lr=1e-2)
orig_poZ = bm.poZ_body.clone()

render = pyredner.RenderFunction.apply
for t in range(NUM_ITERS):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image

    shape_triangle.vertices = bm.forward().v[0]
    scene = pyredner.Scene(cam, shapes, materials, area_lights)
    scene_args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4 , # We use less samples in the Adam loop.
        max_bounces = 1)
    img = render(t+1, *scene_args).double()
    # print(img)

    # Save the intermediate render.
    # if t % 10 == 0:
    pyredner.imwrite(img.cpu(), 'results/' + MODEL_NAME + '/iter_{}.png'.format(t))
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).mean()
    # diff = orig_poZ.double() - bm.poZ_body.double()
    # loss += (diff.pow(2).mean())
    loss.backward()
    print('Loss:', loss.item())
    # Take a gradient descent step.
    optimizer.step()

    # Pickle the poZ_body
    pickle_out = open('results/' + MODEL_NAME + '/poZ_body.pkl',"wb")
    pickle.dump(dict(body_pose=bm.poZ_body), pickle_out)
    pickle_out.close()

    # Save the resulting 3D model in .obj format
    result_obj = shapes[0]
    result_obj.vertices = result_obj.vertices.detach()
    pyredner.save_obj(result_obj, f'results/' + MODEL_NAME + '/result.obj')

# Save the images and differences.
pyredner.imwrite(img.cpu(), 'results/' + MODEL_NAME + '/final.exr')
pyredner.imwrite(img.cpu(), 'results/' + MODEL_NAME + '/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/' + MODEL_NAME + '/final_diff.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(['ffmpeg', '-framerate', '24', '-i',
    'results/' + MODEL_NAME + '/iter_%d.png', '-vb', '20M',
    'results/' + MODEL_NAME + '/out.mp4'])
