import pyredner
import torch
import matplotlib.pyplot as plt
import numpy as np
from human_body_prior.body_model.body_model_vposer import BodyModelWithPoser

# Get the body model
expr_dir = '../SMPL-X/vposer_v1_0' #'TRAINED_MODEL_DIRECTORY'  in this directory the trained model along with the model code exist
bm_path =  '../SMPL-X/models/smplx/SMPLX_MALE.npz'#'PATH_TO_SMPLX_model.npz'  obtain from https://smpl-x.is.tue.mpg.de/downloads
bm = BodyModelWithPoser(bm_path=bm_path, batch_size=1, poser_type='vposer', smpl_exp_dir=expr_dir)

# HERE WAS AN ATTEMPT TO START FROM THE GOOD APPROXIMATION
# pose_body = [ 0.08749139, -1.030376  , -0.7968941 , -0.59646785, -2.395034  ,
#   -0.38647613,  1.7464787 ,  2.322542  , -0.05406742,  0.3268399 ,
#    0.11413634, -0.12440143,  0.22781989,  1.9553049 ,  1.367039  ,
#   -0.905784  ,  0.07017767, -1.3596315 , -0.4142471 ,  0.4299623 ,
#    1.7596375 ,  0.28724512, -2.496917  ,  1.1127855 ,  0.17023656,
#   -0.7269698 ,  0.33199713, -0.85020965, -0.06031673, -0.0870313 ,
#   -2.131748  , -0.28954616]

 
# bm.forward(poZ_body = torch.tensor(pose_body, requires_grad=True))



# Prepare to render
target_img = pyredner.imread('kouchi_photo_tori.jpg')
target = torch.tensor(target_img).double()
pyredner.imwrite(target.cpu(), 'results/optimize_pose_kouchi_tori_custom_pose/target.png')

img_dim = (target.shape[0], target.shape[1])

cam_pos = torch.tensor([-0.1, -0.4, 1.4], requires_grad=True)
cam = pyredner.Camera(position = cam_pos,
                      look_at = torch.tensor([-0.1, -0.4, 0.0]),
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
optimizer = torch.optim.Adam([bm.poZ_body], lr=1)
cam_optimizer = torch.optim.Adam([cam_pos])

render = pyredner.RenderFunction.apply
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    cam_optimizer.zero_grad()
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
    if t % 10 == 0:
      pyredner.imwrite(img.cpu(), 'results/optimize_pose_kouchi_tori_custom_pose/iter_{}.png'.format(t))
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).mean()
    # loss += 0.1 * torch.norm(bm.poZ_body).double()      
    loss.backward(retain_graph=True)
    print('Loss:', loss.item())
    # Take a gradient descent step.
    optimizer.step()
    cam_optimizer.step()
