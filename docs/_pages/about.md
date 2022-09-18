---
layout: defaultPaper
title: Combining Motion Matching and Orientation Prediction to Animate Avatars for Consumer-Grade VR Devices
permalink: /
---

<center>
<figure style="display:inline-block;margin:0;padding:0"><img src="assets/img/teaser.jpg" alt="teaser" width="100%" /><figcaption style="text-align:center"></figcaption></figure>
</center>

<br>

<div class="img_horizontal_container">
	<a href="assets/pdf/motion_matching_vr.pdf">
	<div class="img-with-text">
		<img src="assets/img/article_icon.svg" alt="paper" />
		<p><b>Paper</b></p>
	</div>
	</a>
	<a href="https://github.com/UPC-ViRVIG/MMVR">
	<div class="img-with-text">
		<img src="assets/img/github_icon.svg" alt="code" />
		<p><b>Code</b></p>
	</div>
	</a>
	<a href="https://github.com/UPC-ViRVIG/MMVR#data">
	<div class="img-with-text">
		<img src="assets/img/database_icon.svg" alt="data" />
		<p><b>Data</b></p>
	</div>
	</a>
</div>

------

<h3><center><b>
Abstract
</b></center></h3>

<div style="text-align: justify;">
The animation of user avatars plays a crucial role in conveying their pose, gestures, and relative distances to virtual objects
or other users. Consumer-grade VR devices typically include three trackers: the Head Mounted Display (HMD) and
two handheld VR controllers. Since the problem of reconstructing the user pose from such sparse data is ill-defined,
especially for the lower body, the approach adopted by most VR games consists of assuming the body orientation matches
that of the HMD, and applying animation blending and time-warping from a reduced set of animations. Unfortunately, this
approach produces noticeable mismatches between user and avatar movements. In this work we present a new approach to
animate user avatars for current mainstream VR devices. First, we use a neural network to estimate the user’s
body orientation based on the tracking information from the HMD and the hand controllers. Then we use this orientation
together with the velocity and rotation of the HMD to build a feature vector that feeds a Motion Matching algorithm. We built a
MoCap database with animations of VR users wearing a HMD and used it to test our approach on both self-avatars and other
users’ avatars. Our results show that our system can provide a large variety of lower body animations while correctly matching
the user orientation, which in turn allows us to represent not only forward movements but also stepping in any direction.
</div>

<div class="row">
  <div class="column">
    <figure style="display:inline-block;margin:0;padding:0"><img src="assets/img/dancing.gif" alt="dancing avatar" width="100%" /><figcaption style="text-align:center"></figcaption></figure>
  </div>
  <div class="column">
	<figure style="display:inline-block;margin:0;padding:0"><img src="assets/img/first_person.gif" alt="first-person view" width="100%" /><figcaption style="text-align:center"></figcaption></figure>
  </div>
</div>

--------

<br>

<h3><b>
Method
</b></h3>
We propose a new method to animate self-avatars using only one HMD and two hand-held controllers.
Our system can be divided into three parts: 

- Body orientation prediction
- Motion Matching
- Final pose adjustments

<center>
<figure style="display:inline-block;margin:0;padding:0"><img src="assets/img/pipeline.png" alt="pipeline" width="100%" /><figcaption style="text-align:center"></figcaption></figure>
</center>

<br>

<div style="background-color:rgba(244, 251, 255, 1.0); vertical-align: middle; padding:10px 20px; text-align: justify;">
<h3><b>
Body Orientation Prediction
</b></h3>
Predicting the body orientation is a common problem in applications using full-body avatars
with only one HMD and two controllers. Current methods use the HMD's forward direction to orient the whole body,
producing mismatches with the actual body orientation.
Instead, we trained a lightweight feedforward neural network to predict the body orientation from the rotation,
velocity and angular velocity of all three devices.

<center>
<figure style="display:inline-block;margin:0;padding:0"><img src="assets/img/nn.png" alt="orientation prediction neural network" width="75%" /><figcaption style="text-align:center"></figcaption></figure>
</center>

Predicting the orientation directly from the ground truth data would not match the real
usage scenario of the network, and therefore, the network would not be learning how
to predict the next orientation based on the previously predicted one.
Instead, for every element in a training batch, we iteratively predict the orientation
\( r \) times (e.g., \( r=50 \) ). Then, we compute the MSE loss by comparing the final predicted
body orientation \( \mathbf{\hat{d}} \) with the ground truth orientation \( \mathbf{d^*} \) after \( r \) frames.
</div>

<br>

<div style="background-color:rgba(255, 247, 247, 1.0); vertical-align: middle; padding:10px 20px; text-align: justify;">
<h3><b>
Motion Matching for VR
</b></h3>
Motion Matching searches over an animation database for the best match for the current avatar
pose and the predicted trajectory.
To find the best match, we compute a new database with the main features defining locomotion. 
A feature vector \( \mathbf{z} \in \mathbb{R}^{27} \) is defined for each pose. This feature vector combines
two types of information: the current pose and the trajectory. When comparing feature vectors,
the former ensures no significant changes in the pose and thus smooth transitions;
the latter drives the animation towards our target trajectory. Feature vectors are defined as follows:
\begin{equation*}
    \mathbf{z} = \left( \mathbf{z^v}, \mathbf{z^l}, \mathbf{z^p}, \mathbf{z^d} \right)
\label{eq:z}
\end{equation*}
where \(  \mathbf{z^v}, \mathbf{z^l} \) are the current pose features and \(  \mathbf{z^p}, \mathbf{z^d} \) 
are the trajectory features. More precisely, \( \mathbf{z^v} \in \mathbb{R}^{9} \) are the velocities
of the feet and hip joints, \( \mathbf{z^l} \in \mathbb{R}^{6} \) are the positions of the feet joints,
\( \mathbf{z^p} \in \mathbb{R}^{6} \) and \( \mathbf{z^d} \in \mathbb{R}^{6} \) are the future 2D positions
and 2D orientations of the character \( 0.33 \) , \( 0.66 \) and \( 1.00 \) seconds ahead.
<center>
<figure style="display:inline-block;margin:10px;padding:0"><img src="assets/img/features.gif" alt="features" width="75%" /><figcaption style="text-align:center"></figcaption></figure>
</center>

</div>

<br>

<div style="background-color:rgba(255, 252, 243, 1.0); vertical-align: middle; padding:10px 20px; text-align: justify;">
<h3><b>
Final pose adjustments
</b></h3>

In our work, the upper body is not considered for the Motion Matching algorithm to avoid increasing the
dimensionality of the feature vector and focus instead on lower body locomotion, for which no tracking
data is available in consumer-grade VR. 
In order to obtain the upper body pose for the arms, we can use the hand controllers as end effectors
for an Inverse Kinematics algorithm. This solution is fast to compute and provides a good solution
for the user to interact with the environment in VR.

<center>
<figure style="display:inline-block;margin:10px;padding:0"><img src="assets/img/upper_ik.gif" alt="upper body IK" width="30%" /><figcaption style="text-align:center"></figcaption></figure>
</center>

</div>

<br>

-----

<h3><center><b>
Overview Video
</b></center></h3>

<center>
<div class="video_wrapper">
	<iframe frameborder="0" width="100%" height="100%"
	src="https://www.youtube.com/embed/crU9oLX0GnM">
	</iframe>
</div>
</center>

-----

<br>

<h3><b>
Citation
</b></h3>
<div style="background-color:rgba(0, 0, 0, 0.03); vertical-align: middle; padding:10px 20px;">
@article {ponton2022mmvr, <br>
	journal = {Computer Graphics Forum}, <br>
	{% raw %}
	title = {{Combining Motion Matching and Orientation Prediction to Animate Avatars for Consumer-Grade VR Devices}}, <br>
	{% endraw %}
	author = {Ponton, Jose Luis and Yun, Haoran and Andujar, Carlos and Pelechano, Nuria}, <br>
	year = {2022}, <br>
	publisher = {The Eurographics Association and John Wiley & Sons Ltd.}, <br>
	ISSN = {1467-8659}, <br>
	DOI = {10.1111/cgf.14628} <br>
}
</div>