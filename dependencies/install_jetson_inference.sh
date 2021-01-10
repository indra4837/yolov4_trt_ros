#!/bin/bash

cd ~/catkin_ws/src/jetson-inference
mkdir build
cd build
sudo cmake ../
sudo make -j$(nproc)
sudo make install
sudo ldconfig



