## Color Segmentation using HSV color space

This is code to segment regions in an image based on the color values of any given training images in the HSV space. Following are the steps to execute -  

Create an instance of the class ColorSegmentation with the directory path as parameter to the constructor. You can use regular expressions to retrieve the training images of interest -
```python
obj = ColorSegmentation('./training images/*.jpg')
```

#### Histogram creation and plotting

To build the histogram, run the create_histogram() function along with the name of the file to which the scatter plot must be saved. The histogram is saved in self.histogram and the plot is saved to the path provided -
```python
obj.create_histogram('plot.png')
```

#### Detect skin

To detect skin in the input image, pass the input image along with the destination path and proportion -
```python
obj.extract_skin('input.png', 'output.png', 0.75)
``` 