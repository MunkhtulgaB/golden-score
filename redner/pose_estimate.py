import pyredner
import torch


target_img = pyredner.imread('kouchi_photo_tori_small.jpg')
target = torch.tensor(target_img)
img_dim = (target.shape[0], target.shape[1])

objects = pyredner.load_obj('kouchi_tori.obj', return_objects=True)
camera = pyredner.automatic_camera_placement(objects, resolution=img_dim)

# Define model that takes objects, camera and pose params and outputs an image
vertices = []
for obj in objects:
	vertices.append(obj.vertices.clone())

center = torch.mean(torch.cat(vertices), 0)
def model(translation, euler_angles):
	rotation_matrix = pyredner.gen_rotate_matrix(euler_angles)
	for obj, v in zip(objects, vertices):
		obj.vertices = (v - center) @ torch.t(rotation_matrix) + center + translation

	scene = pyredner.Scene(camera = camera, objects = objects)
	img = pyredner.render_albedo(scene)
	return img


# Target image
# target_translation = torch.tensor([0.0, 0.0, 0.0], device = pyredner.get_device())
# target_euler_angles = torch.tensor([0.0, 0.0, 0.0], device = pyredner.get_device())
# target = model(target_translation, target_euler_angles).data


# Show
import matplotlib.pyplot as plt
# plt.imshow(torch.pow(target, 1.0/2.2).cpu())
# plt.show()

# Initial guess
# Set requires_grad=True since we want to optimize them later
translation = torch.tensor([0.0, 0.0, 0.0], device = pyredner.get_device(), requires_grad=True)
euler_angles = torch.tensor([0.1, -0.1, 0.1], device = pyredner.get_device(), requires_grad=True)
init = model(translation, euler_angles)
# Visualize the initial guess
plt.imshow(torch.pow(init.data, 1.0/2.2).cpu()) # add .data to stop PyTorch from complaining
plt.show()


# optimise the pose
# t_optimizer = torch.optim.Adam([translation], lr=0.5)
r_optimizer = torch.optim.Adam([euler_angles], lr=0.01)


# Set up plotting
import matplotlib.pyplot as plt
# %matplotlib inline
from IPython.display import display, clear_output
import time
plt.figure()
imgs, losses = [], []
# Run 80 Adam iterations
num_iters = 10
for t in range(num_iters):
    # t_optimizer.zero_grad()
    r_optimizer.zero_grad()
    img = model(translation, euler_angles)
    # Compute the loss function. Here it is L2.
    # Both img and target are in linear color space, so no gamma correction is needed.
    loss = (img - target).pow(2).mean()
    loss.backward()
    print('loss', loss)
    print()
    # t_optimizer.step()
    r_optimizer.step()
    print(euler_angles)
    # Plot the loss
    # if t == num_iters - 1:
    f, (ax_loss, ax_img) = plt.subplots(1, 2)
    losses.append(loss.data.item())
    imgs.append(torch.pow(img.data, 1.0/2.2).cpu()) # Record the Gamma corrected image
   #  clear_output(wait=True)

ax_loss.plot(range(len(losses)), losses, label='loss')
ax_loss.legend()
ax_img.imshow((img-target).pow(2).sum((2)).data.cpu())
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
