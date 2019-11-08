## Canny Edge Detection

This is code that implements canny edge detection from scratch. The following are the steps to execute -

Create an instance of the class CannyEdgeDetection with the image path as parameter to the constructor -
```python
img = CannyEdgeDetection('example.png')
```

#### Gaussian smoothing

To smooth the image, run the gaussian_smoothing function with kernel size and sigma as parameters -
```python
smooth = img.gaussian_smoothing((3,3), 5)
```

#### Image gradients

To find the gradient magnitude and direction, run the image_gradient function on the smoothed image -
```python
mag, theta = img.image_gradient(smooth)
``` 

#### Non-maximum suppression

To suppress non-maximum pixels, pass the gradient magnitude and gradient direction to the non_maximum_suppression -
```python
supp = img.non_maximum_suppression(mag, theta)
```

#### Thresholding 

To perform thresholding, pass the suppressed image as well as a threshold value to the threshold function -
```python
th = img.threshold(supp)
```

#### Edge linking

To link the weak edges to the strong ones, pass the thresholded image to the edge_link function -
```python
link = img.edge_link(th)
```

link is the final image in the step. It is a binary image that produces the final edges chosen. 