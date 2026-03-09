# Dataset tools


## FetchBench Dataset revise

Post-processing for the OBJ dataset obtained from ShapeNet, and scripts for adding scenes with joints to FetchBench’s Isaac Gym.

## Before start
#### OBJ
OBJ refers to both the target objects and the obstacle objects collectively.

#### SCENE
Scene refers to environmental objects where the OBJ can be placed, such as tables, shelves, drawers, cabinets, etc.



## Scripts related to OBJ
### 1. resize.py
A script to resize the OBJ dataset obtained from ShapeNet to the scale used in FetchBench.
### 2. check_item_jpgs.py
A script to transfer the texture images from the OBJ dataset obtained from ShapeNet to the FetchBench OBJ dataset.


## Scripts related to SCENE
### 1. add_collision_to_urdf (Temporary)
A script to add collision elements to the URDF files of the scene dataset generated with Infinigen.
### 2. save_metadata_npy.py
A script to generate metadata.npy.
### 3. visualize_support_polygon.py
A script to visually inspect the JSON file that defines the OBJ placement positions in the scene dataset generated with Infinigen.