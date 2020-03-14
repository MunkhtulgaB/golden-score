import re
import sys
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument(
	'--tori', 
	'-t', 
	help=".obj file containing 3D model of tori", 
	type=str,
	default="tori.obj"
)
parser.add_argument(
	'--uke', 
	'-u', 
	help=".obj file containing 3D model of uke", 
	type=str, 
	default="uke.obj"
)
parser.add_argument(
	'--output', 
	'-o', 
	help="name for file to write merged the model", 
	type=str, 
	default="grip.obj"
)

args = parser.parse_args()
print(args)

tori = open(args.tori, "r")
uke = open(args.uke, "r")

grip = open(args.output, "w")

# Read vertices and faces from tori file
tori_vertices = []
tori_faces = []
while True:
    nextLine = tori.readline()
    if not nextLine:
        break
    else:
    	if nextLine.startswith('v'):
    		tori_vertices.append(nextLine)
    	elif nextLine.startswith('f'): 
    		tori_faces.append(nextLine)

# 1. Write the tori vertices
grip.writelines(tori_vertices)

def translate(orig, translation):
	x, y, z = orig
	dx, dy, dz = translation
	return (x + dx, y + dy, z + dz)

def rotate(orig, pivot, angle, axis=0):
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

translation = (-0.1, 0.22, -0.5)
pivot = (0,0,0)
rotation_angle = -(math.pi / 2) * 0.4
rotation_axis = 0

rotation_angle1 = (math.pi / 2) * 1.8
rotation_axis1 = 1

rotation_angle2 = (math.pi / 2) * 0.35
rotation_axis2 = 2

# Read vertices and faces from uke file
uke_vertices = []
uke_faces = []
while True:
    nextLine = uke.readline()
    if not nextLine:
        break
    else:
    	if nextLine.startswith('v'):
    		tokens = nextLine.rstrip().split(" ")[1:]
    		x, y, z = [float(t) for t in tokens]

    		
    		
    		tokens[0], tokens[1], tokens[2] = (str(c) for c in \
                translate(
                    rotate(
                        rotate(
                            rotate( (x, y, z) , pivot, rotation_angle, axis=rotation_axis ),
                            pivot,
                            rotation_angle1,
                            rotation_axis1
                        ),
                        pivot,
                        rotation_angle2,
                        rotation_axis2
                    ),
                    translation
                )
            )
    		new = " ".join(['v'] + tokens)

    		uke_vertices.append(new + '\n')
    	elif nextLine.startswith('f'):
    		offset = len(tori_vertices)
    		new = re.sub('(\d+)', lambda m: str(int(m.group(1)) + offset), nextLine)
    		uke_faces.append(new)

# 2. Write uke vertices
# 3. Write tori faces
# 4. Write uke faces
grip.writelines(uke_vertices)
grip.writelines(tori_faces)
grip.writelines(uke_faces)

print('Gripped model written to:', args.output)

