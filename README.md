# Smiley_SVG

## Requirements: 

- python >= 3.7.10
- librairies: numpy, torch, torchvision, drawSvg, svgwrite, matplotlib 

## Description: 

The aim of this project is to generate an image of a smiley in vector format (such as Scalable Vector Graphics) form a vector of float numbers specifying the positions of the eyes and the curvature of the mouth. Two models were implemented: VAE_smiley and cVAE_smiley, corresponding respectively to a Variational AutoEncoder (VAE) and a conditional Variational AutoEncoder (cVAE). As I used ”DrawSVG”, a Python 3 library for programmatically generating SVG images and displaying them in a Jupyter notebook, I have provided two Jupyter notebooks that can be run directly with saved trained models to visualise the results.

__Dataset:__

A smiley can easily be represented by a vector of attributes. In our data set, each smiley corresponds to a vector of size 10: [eye1x, eye1y, eye2x, eye2y, m1x, m1y, m2x, m2y, m3x, m3y]. The tuples (eye1x, eye1y) and (eye2x, eye2y) specify the position of each eye whilst the other coefficients are used to draw the mouth, which corresponds to the curve passing through the three points of coordinates (m1x, m1y), (m2x, m2y), and (m3x, m3y). As we have displayed each smiley on a canvas of size 100 × 100, where we define the center as the origin, each coefficient was a float number with a value in  [−50,50].


