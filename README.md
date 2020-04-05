# Golden Score - capturing the golden moment in judo!

Tension builds up as a judoka warms up silently. Across the hall is his opponent with whom the judoka trades secret glances. They both have paid their sweat, blood and tears to see each other on the mat today. 

Finally, it's their turn. Life may go by slowly and steadily sometimes, but the following five or so minutes will be full of uncertainties and thrills that are enough to beat any theme park. Bow, and begin! Both fighters are now finally allowed to unleash their billions years worth of survival instincts, and the millennia old wisdom of warriors passed down to them. 

After seconds of fighting that will burn into the fighters' memories more than months of idle-sitting, all the uncertainties, nerves, winning or losing will clear in a single moment of grace. This is a single moment in which a world-champion will be crowned. This is a single moment in which history tips to its course. This is the **the GOLDEN moment of JUDO!**

This is my individual project to capture those moments using state-of-the-art computer vision technologies.

<img src="https://github.com/MunkhtulgaB/golden-score/blob/master/finished_models/gold_images/osotogari.obj.png" alt="Osotogari by Munkhtulga Battogtokh"/> 
<img src="https://github.com/MunkhtulgaB/golden-score/blob/master/finished_models/gold_images/tomoenage.obj.png" alt="Tomoenage by Shohei Ono"/> 
<img src="https://github.com/MunkhtulgaB/golden-score/blob/master/finished_models/gold_images/osoto_fonseca.obj.png" alt="Osotogari by Fonseca Jorge"/>

### Dependencies
* **SMPLify-x** Expressive Body Capture: 3D Hands, Face, and Body from a Single Image (https://github.com/vchoutas/smplify-x)
* **redner**: Differentiable rendering without approximation (https://github.com/BachiLi/redner)
* **OpenPose** (https://github.com/CMU-Perceptual-Computing-Lab/openpose)
* **Grabcut** (https://github.com/opencv/opencv/blob/master/samples/python/grabcut.py)

### Instructions

Note that the process may require you to do complicated time-consuming environment set up. All information about how to set up the environment for each dependency is available with their source code on Github (see Dependencies). Refer to their respective Github repositories and documentations I provide here if you wish to replicate the results.


1. Pick input image(s)
2. a. Pre-process with Grabcut. Follow the sample provided on OpenCV Github repository (https://github.com/opencv/opencv/blob/master/samples/python/grabcut.py). Alternatively, use slightly modified (made interaction simpler) version of this sample within this repository in `Grabcut/grabcut.py`.
   
   b. Then place your input image in the same directory and run:
      ```
      python grabcut.py <input image>
      ```
    to start the interactive foreground detection. Reduce image to one fighter only; repeat for each fighter so you will have 2 images at the end of this step. Tested environment: Windows 10.
3. a. Generate keypoints using OpenPose for each of the 2 images you have (see https://github.com/CMU-Perceptual-Computing-Lab/openpose). Tested environment: Windows 10.
   
   b. Construct the model using SMPLify-x (see dependencies) by setting up the enivronment and running:
      ```
       python3 smplifyx/main.py --config cfg_files/fit_smplx.yaml --data_folder <input directory> --output_folder <output directory> --model_folder <model folder> --gender male 
      ```
      At the end of this step, in the output directory you will have `meshes` where Wavefront OBJ 3D models of the fighters are created. You will also have .pkl files from which you can recover the model objects in the `results` subdirectory. See tutorial on SMPLify-x Github repository for any details (https://github.com/vchoutas/smplify-x). Tested environment: MacOS Mojave
        
4. (Optional) Optimise each individual model using:

   a. An alternative pose prior distribution. Add the following directory under the directory you installed SMPLify-x and name it vposer_judo:
           ```
           https://github.com/MunkhtulgaB/golden-score/tree/master/vposer_training/training/judo
           ```
        To use this pose prior distribution instead of the default VPoser distribution (which is named vposer_v1_0 by default), change the following line in `<installation folder>/smplify-x/cfg-files/fit-smplx.yaml`: 
           ```
           vposer_ckpt: "../vposer_v1_0"
           ```
           with 
           ```
           vposer_ckpt: "../vposer_v1_0"
           ```

     b. Differential Rendering optimisation. The source code to optimise pose of a fighter is provided in:
           ```
           https://github.com/MunkhtulgaB/golden-score/blob/master/vposer/optimize_pose.py
           ```
      Download and change `MODEL_NAME` constant to the name of the model `<input>` you wish to optimise. Then place the .pkl file    from the `results` subdirectory mentioned in step 3b in: 
           ```
           ./input/source/<input>/
           ```
           Also place the target images you wish to optimise against in:
           ```
           ./input/targets/
           ```
           Then simply run with: 
           ```
           python3 optimize_pose.py
           ``` 
    At the end of this step, you will have modified versions of the two 3D models for each fighter.
    
5. Merge the individual 3D models together. Do this using (any 3D editor if you are boring) or by first downloading:
    ```
    https://github.com/MunkhtulgaB/golden-score/blob/master/gripping/grip.py
    ```
    
    Then place the two models in the same directory, and run with:
    
    ```
    python3 grip.py --tori <tori's OBJ model file> --uke <uke's OBJ model file>
    ```
    
    Upon success, it will print: 
    ```
    Gripped model written to: grip.obj
    ```
    
    The final model is then now output as `grip.obj`
    
    **Note** You must manually configure the necessary translation and rotation of the uke by changing the following lines:
    ```
    translation = (-0.4, 0.05, 0.15)
    pivot = (0,0,0)
    rotation_angle = -math.pi/2
    rotation_axis = 1

    rotation_angle1 = math.pi/7
    rotation_axis1 = 0

    rotation_angle2 = 0
    rotation_axis2 = 2
    ```
    
    This will require some experimentation, and knowledge of transformations in 3D space. But this is intuitive, and often trivial if you experiment starting from setting the rotation angles to `math.pi/2` (90 degrees).
