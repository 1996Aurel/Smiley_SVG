# Smiley_SVG

## Requirements: 

- python >= 3.7.10
- librairies: numpy, torch, torchvision, drawSvg, svgwrite, matplotlib 

## Description: 

The aim of this project is to generate an image of a smiley in vector format (such as Scalable Vector Graphics) form a vector of float numbers specifying the positions of the eyes and the curvature of the mouth. Two models were implemented: VAE_smiley and cVAE_smiley, corresponding respectively to a Variational AutoEncoder (VAE) and a conditional Variational AutoEncoder (cVAE). As we used ”DrawSVG”, a Python 3 library for programmatically generating SVG images and displaying them in a Jupyter notebook, we have provided two Jupyter notebooks that can be run directly with saved trained models to visualise the results.

__Dataset:__

A smiley can easily be represented by a vector of attributes. In our data set, each smiley corresponds to a vector of size 10: [eye1x, eye1y, eye2x, eye2y, m1x, m1y, m2x, m2y, m3x, m3y]. The tuples (eye1x, eye1y) and (eye2x, eye2y) specify the position of each eye whilst the other coefficients are used to draw the mouth, which corresponds to the curve passing through the three points of coordinates (m1x, m1y), (m2x, m2y), and (m3x, m3y). As we have displayed each smiley on a canvas of size 100 × 100, where we define the center as the origin, each coefficient was a float number with a value in  [−50,50].


To create our data set of smileys, we have implemented two functions: generate_matrix() and generate matrix_cond(y). The first function returned a random vector of size ten corresponding to a smiley where the eyes were located in the upper part of the canvas, while the mouth was always placed under
the eyes. Moreover, the smileys generated were symmetric about the y axis. The second function was similar to the first one, except that we introduced the binary condition y. If y = 0, then the function returned a sad smiley, otherwise, if y = 1, it returned a happy smiley. Finally, we created a dataset of 7,000 smileys where 5,000 were used for the training, 1,000 for validation, and 1,000 for testing.

__Models:__

blabla 
![image](https://user-images.githubusercontent.com/121833780/212603688-dc758367-cc81-4487-b069-fe7b823d7d45.png)


blublu
![image](https://user-images.githubusercontent.com/121833780/212603716-d02fb1c7-a8d6-499e-8dad-20d173f56e82.png)




