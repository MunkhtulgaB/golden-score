import pyredner
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
from human_body_prior.body_model.body_model_vposer import BodyModelWithPoser


NUM_ITERS = 50
MODEL_NAME = 'tori_osotogari'

# Get the body model
expr_dir = '../SMPL-X/vposer_v1_0' #'TRAINED_MODEL_DIRECTORY'  in this directory the trained model along with the model code exist
bm_path =  '../SMPL-X/models/smplx/SMPLX_MALE.npz'#'PATH_TO_SMPLX_model.npz'  obtain from https://smpl-x.is.tue.mpg.de/downloads
bm = BodyModelWithPoser(bm_path=bm_path, batch_size=1, poser_type='vposer', smpl_exp_dir=expr_dir)

# Start from an approximation loaded from pickled SMPLify-x output
body_model_path = 'input/source/' + MODEL_NAME + '/'
body_pickle = open(body_model_path + '000.pkl', 'rb')
body = pickle.load(body_pickle)

pose_body = body.get('body_pose')[0]
poZ_body = torch.tensor(pose_body, requires_grad=True)
bm.register_parameter('poZ_body', torch.nn.Parameter(poZ_body, requires_grad=True))


# Prepare to render
target_img = pyredner.imread('input/targets/' + MODEL_NAME + '.png')
target = torch.tensor(target_img).double()
pyredner.imwrite(target.cpu(), 'results/' + MODEL_NAME + '/target.png')

img_dim = (target.shape[0], target.shape[1])

cam_pos = torch.tensor([1., 1., 1.], requires_grad=True)
cam = pyredner.Camera(position = cam_pos,
                      look_at = torch.tensor([0.1, -0.4, 0.0]),
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
    vertices = torch.tensor([[-1.0, -1.0, 3.],
                             [ 1.0, -1.0, 3.],
                             [-1.0,  1.0, 3.],
                             [ 1.0,  1.0, 3.]], device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2],[1, 3, 2]],
        dtype = torch.int32, device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 0)

light = pyredner.AreaLight(shape_id = 1, 
                           intensity = torch.tensor([4.0,4.0,4.0]),
                           two_sided=True,
                           directly_visible=False)

vertices = bm.forward().v[0]
indices = bm.f
shape_triangle = pyredner.Shape(
	vertices = vertices, 
	indices = bm.f, 
	material_id=0)


shapes = [shape_triangle, shape_light]
area_lights = [light]
scene = pyredner.Scene(cam, shapes, materials, area_lights)


# Optimise
optimizer = torch.optim.SGD([bm.poZ_body], lr=1, momentum=0.9, dampening=0.1)
# cam_optimizer = torch.optim.Adam([cam_pos], lr=1e-3)

render = pyredner.RenderFunction.apply
for t in range(NUM_ITERS):
    print('iteration:', t)
    optimizer.zero_grad()
    # cam_optimizer.zero_grad()
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
    # loss += 0.1 * torch.norm(bm.poZ_body).double()      
    loss.backward()
    print('Loss:', loss.item())
    # Take a gradient descent step.
    optimizer.step()
    # cam_optimizer.step()

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
