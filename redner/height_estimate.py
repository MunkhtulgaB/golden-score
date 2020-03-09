import pyredner
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math


target = torch.tensor(mpimg.imread('target/moonstep.jpg')).double()
img_dim = (target.shape[0], target.shape[1])


objects = pyredner.load_obj('mori_knob/testObj.obj', return_objects=True)
objects = [objects[0], objects[3]]
camera = pyredner.automatic_camera_placement([objects[0]], resolution=img_dim)

# Define pose that takes objects, camera and pose params and outputs an image
vertices = []
for obj in objects:
	vertices.append(obj.vertices.clone())

center = torch.mean(torch.cat(vertices), 0)
def pose(translation, euler_angles):
	rotation_matrix = pyredner.gen_rotate_matrix(euler_angles)
	for obj, v in zip(objects, vertices):
		obj.vertices = (v - center) @ torch.t(rotation_matrix) + center + translation

	scene = pyredner.Scene(camera = camera, objects = objects)
	img = pyredner.render_albedo(scene)
	return img.double()

def model(heights):
    obj.vertices *= 2 

    scene = pyredner.Scene(camera = camera, objects = objects)
    img = pyredner.render_albedo(scene)
    return img.double()

# target_translation = torch.tensor([0.0, 0.0, 0.0], device = pyredner.get_device())
# target_euler_angles = torch.tensor([0.0, 0.0, 0.0], device = pyredner.get_device())
# target = pose(target_translation, target_euler_angles).data
# print(target)

# Show
# plt.imshow(torch.pow(target, 1.0/2.2).cpu())
# plt.show()

# Initial guess
# Set requires_grad=True since we want to optimize them later
translation = torch.tensor([0.0, 0.0, -15.0], device = pyredner.get_device(), requires_grad=True)
euler_angles = torch.tensor([math.pi/2, 0., 0.], device = pyredner.get_device(), requires_grad=True)
init = pose(translation, euler_angles)
img = init

heights = objects[0]
print('-----------------------------')
print(heights)
# Visualize the initial guess
# plt.imshow(torch.pow(init.data, 1.0/2.2).cpu()) # add .data to stop PyTorch from complaining
# plt.show()


# optimise the pose
# t_optimizer = torch.optim.Adam([translation], lr=0.5)
# r_optimizer = torch.optim.Adam([euler_angles], lr=0.01)
h_optimizer = torch.optim.Adam([heights], lr=0.1)


# Set up plotting
import matplotlib.pyplot as plt
# %matplotlib inline
from IPython.display import display, clear_output
import time
plt.figure()
imgs, losses = [], []
# Run 80 Adam iterations
num_iters = 1
for t in range(num_iters):
    t_optimizer.zero_grad()
    r_optimizer.zero_grad()
    
    # print(img)
    # print(target)
    # Compute the loss function. Here it is L2.
    # Both img and target are in linear color space, so no gamma correction is needed.
    loss = (img - target).pow(2).mean()
    loss.backward()
    t_optimizer.step()
    r_optimizer.step()
    # Plot the loss
    # if t == num_iters - 1:
    f, (ax_loss, ax_img, ax_source) = plt.subplots(1, 3)
    losses.append(loss.data.item())
    imgs.append(torch.pow(img.data, 1.0/2.2).cpu()) # Record the Gamma corrected image
   #  clear_output(wait=True)
print(img)
ax_loss.plot(range(len(losses)), losses, label='loss')
ax_loss.legend()
ax_img.imshow((img-target).pow(2).sum((2)).data.cpu())
ax_source.imshow(torch.pow(img, 1.0/2.2).data.cpu())

plt.show()

from matplotlib import animation
from IPython.display import HTML
fig = plt.figure()
im = plt.imshow(imgs[0], animated=True)
def update_fig(i):
    im.set_array(imgs[i])
    return im,
anim = animation.FuncAnimation(fig, update_fig, frames=len(imgs), interval=50, blit=True)

from IPython.display import HTML
HTML(anim.to_jshtml())
