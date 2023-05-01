# Stereo Disparity Estimation

- based on stereo disparity estimation algorithm proposed by Stereo Hybrid Event-Frame (SHEF) Cameras for 3D Perception

- a semester project on event-based robot vision by Nan Cai, supervised by Prof. Dr. Guillermo Gallego. 
- The report explores the use of event and frame cameras for disparity estimation 

![A block diagram of the stereo disparity estimation](images/overview.png)

## Result

<p align="left">
   <img src="images/illustration_warping.png" alt="An illustration of image warping" width="400"/>
</p>

We use image warping process in above to evaluate the quality of the estimated disparity map

<p align="left">
   <img src="images/imfuse2_8.png" alt="The results of image warping" width="400"/>
</p>
The right image (event, marked in green) is warped to the
left (frame, marked in purple) image