#!/bin/bash


for idx in 16 18 20 21 23 27 40 41 45 46
do
  for i in {1..10}
  do
    python generate_scenes.py headless=True task.env.sceneCategory=CellShelfDeskSceneFactory task.env.sceneIdx=[$idx]
  done
done

for idx in 3 11 25 30 46 48
do
  for i in {1..10}
  do
    python generate_scenes.py headless=True task.env.sceneCategory=DeskSceneFactory task.env.sceneIdx=[$idx]
  done
done

for idx in 3 12 14 18 22 23 25 26 32
do
  for i in {1..10}
  do
    python generate_scenes.py headless=True task.env.sceneCategory=DeskWallSceneFactory task.env.sceneIdx=[$idx]
  done
done
