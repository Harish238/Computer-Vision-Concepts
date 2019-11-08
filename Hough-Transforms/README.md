## Line Detection using Hough Lines

This is code to implement Hough Transforms from scratch. Following are the steps to execute -

Create an instance of the class HoughTransforms with the image path as parameter to the constructor -
```python
img = HoughTransforms('example.png')
```

#### Canny edge detection

To retrieve the binary image with edges, run the Canny function with high_threshold as parameter. Remember, low_threshold is 50% of high_threshold -
```python
edges = img.canny(75)
```

#### Generate accumulator

To generate an empty accumulator array, run the generate_accumulator function with accumulator height and width as the parameters -
```python
acc = img.generate_accumulator(180, 180)
``` 

#### Voting

To accumulate votes in the accumulator array, pass the binary edge image and accumulator to the accumulate function -
```python
acc1 = img.accumulate(edges, acc)
```

#### Local maxima 

To retrieve the final list of lines, pass the accumulated array as well as a voting threshold to the find_maxima function -
```python
lines = img.find_maxima(acc1, 60)
```

#### Draw the lines

To draw the lines on the original image, pass the derived lines along with the filename.png to write to -
```python
img.draw(lines, filename)
``` 