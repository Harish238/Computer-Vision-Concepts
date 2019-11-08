## Filters

Instantiate an instance of class Filters with the image path as a parameter to the costructor -

```python
img = Filters('example.png')
```

To run a correlation operation on a kernel, call the instance method for correlation with the kernel as a parameter -

```python
filtered_img = img.correlation(some_kernel)
```

Similarly for convolution -

```python
filtered_img = img.convolution(kernel)
```
The kernel is a square, 2D Numpy array with a midpoint.

Median filtering is done using the median() function taking the kernel size as an input -

```python
filtered_img = img.median((5,5))
```

A Gaussian kernel can be generated using the static method gaussian_kernel(). It takes kernel size and sigma as an input to return a Gaussian kernel which can then be used in the convolution and correlation operations -

```python
gauss_kernel = Filters.gaussian_kernel((5,5), 5.0)
filtered_img = img.convolution(kernel)
```
 


