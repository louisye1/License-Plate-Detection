import math
import sys
from pathlib import Path

from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# STUDENT IMPLEMENTATION
# A queue data structure
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

# A useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):
    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array

# This function turns an image into a grayscale image
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height): 
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    # Convert RGB to greyscale
    for i in range(0, image_height):
        for j in range(0, image_width):
            greyscale_pixel_array[i][j] = int(round(0.299*pixel_array_r[i][j] + 0.587*pixel_array_g[i][j] + 0.114*pixel_array_b[i][j]))
    
    return greyscale_pixel_array

# This function stretches the contrast of an image
def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    contrast_stretched_pixel_array  = createInitializedGreyscalePixelArray(image_width, image_height)
    
    # Find min and max values
    min_val, max_val = min([min(i) for i in pixel_array]), max([max(i) for i in pixel_array])
    
    # Scale the contrast
    if min_val == max_val:
        return contrast_stretched_pixel_array 
    else:
        for i in range(0, image_height):
            for j in range(0, image_width):
                contrast_stretched_pixel_array [i][j] = int(round((pixel_array[i][j]-min_val)/(max_val-min_val) * 255))
                
        return contrast_stretched_pixel_array 

# This function filters an image using a 5x5 standard deviation filter
def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):
    filtered_image  = createInitializedGreyscalePixelArray(image_width, image_height)

    # BoundaryIgnore
    for i in range(image_height):
        filtered_image[i][0] = 0.0
        filtered_image[i][image_width-1] = 0.0

    for i in range(image_width):
        filtered_image[0][i] = 0.0
        filtered_image[image_height-1][i] = 0.0

    # Filter the image
    for i in range(2, image_height-2):
        for j in range(2, image_width-2):
            window = pixel_array[i-2][j-2:j+3] + pixel_array[i-1][j-2:j+3] + pixel_array[i][j-2:j+3] + pixel_array[i+1][j-2:j+3] + pixel_array[i+2][j-2:j+3]
            mean = sum(window)/len(window)
            deviations = [(i-mean)**2 for i in window]

            filtered_image[i][j] = (sum(deviations)/len(deviations))**0.5

    return filtered_image

# This function performs a thresholding operation on an image
def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    thresholded = [[0]*image_width for i in range(image_height)]
    
    # Thresholding
    for i in range(0, image_height):
        for j in range(0, image_width):
                if pixel_array[i][j] >= threshold_value:
                    thresholded[i][j] = 255

    return thresholded

# This function erodes an image
def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    eroded_image = createInitializedGreyscalePixelArray(image_width, image_height)
    
    # BorderZeroPadding
    pixel_array.insert(0, [0 for i in range(image_width)])
    pixel_array.append([0 for i in range(image_width)])

    for i in range(0, image_height+2):
        pixel_array[i].insert(0, 0)
        pixel_array[i].append(0)
    
    # Erosion
    for i in range(1, image_height+1):
        for j in range(1, image_width+1):
            window = pixel_array[i-1][j-1:j+2] + pixel_array[i][j-1:j+2] + pixel_array[i+1][j-1:j+2]
            
            if 0 not in window:
                eroded_image[i-1][j-1] = 1
            else:
                eroded_image[i-1][j-1] = 0
                
    return eroded_image

# This function dilates an image
def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    dilated_image = createInitializedGreyscalePixelArray(image_width+2, image_height+2)
    
    # BorderZeroPadding
    pixel_array.insert(0, [0 for i in range(image_width)])
    pixel_array.append([0 for i in range(image_width)])

    for i in range(0, image_height+2):
        pixel_array[i].insert(0, 0)
        pixel_array[i].append(0)
    
    # Dilation
    for i in range(1, image_height+2):
        for j in range(1, image_width+2):
            if pixel_array[i][j] != 0:
                dilated_image[i-1][j-1], dilated_image[i-1][j], dilated_image[i-1][j+1] = 1, 1, 1
                dilated_image[i][j-1], dilated_image[i][j], dilated_image[i][j+1] = 1, 1, 1
                dilated_image[i+1][j-1], dilated_image[i+1][j], dilated_image[i+1][j+1] = 1, 1, 1
                
            elif pixel_array[i][j] == dilated_image[i][j] == 0:
                dilated_image[i][j] = 0
                
    dilated_image = [dilated_image[i][1:image_width+1] for i in range(1, image_height+1)]
    
    return dilated_image

# This function labels the connected components of an image
def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    cc_img = [[0 for i in range(image_width)] for j in range(image_height)]
    cc_sizes = {}
    cc_coordinates = {}
    label = 1
    
    # Label the connected components
    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] != 0 and cc_img[i][j] not in cc_sizes:
                q = Queue()
                q.enqueue((i,j))

                while not q.isEmpty():
                    # Get the coordinates of the current pixel
                    row, col = q.dequeue()
                    # Label the current pixel
                    cc_img[row][col] = label

                    # Add or update the size of the connected component
                    if label not in cc_sizes:
                        cc_sizes[label] = 1
                    cc_sizes[label] += 1

                    # Add the coordinates of the current pixel to the list of coordinates
                    if label not in cc_coordinates:
                        cc_coordinates[label] = []
                    cc_coordinates[label].append((row,col))

                    # Check the neighbors
                    if col-1 >= 0 and pixel_array[row][col-1] != 0 and cc_img[row][col-1] == 0 and (row, col-1) not in q.items:
                        q.enqueue((row,col-1))
                    if col+1 < image_width and pixel_array[row][col+1] != 0 and cc_img[row][col+1] == 0 and (row, col+1) not in q.items:
                        q.enqueue((row,col+1))
                    if row-1 >= 0 and pixel_array[row-1][col] != 0 and cc_img[row-1][col] == 0 and (row-1, col) not in q.items:
                        q.enqueue((row-1,col))
                    if row+1 < image_height and pixel_array[row+1][col] != 0 and cc_img[row+1][col] == 0 and (row+1, col) not in q.items:
                        q.enqueue((row+1,col))
                    
                label += 1

    return cc_img, cc_sizes, cc_coordinates

# This function computes the bounding box of a the largest connected component that has an aspect ratio between 1.5 and 5.0
def computeBoundingBox(cc_coordinates, cc_sizes):
    # Sort the connected components by size, descending
    cc_sizes = dict(sorted(cc_sizes.items(), key=lambda x: x[1], reverse=True))

    # Get the coordinates of the corners of the all connected components
    for i in cc_sizes.keys():
        min_x = min([j[1] for j in cc_coordinates[i]])
        max_x = max([j[1] for j in cc_coordinates[i]])
        min_y = min([j[0] for j in cc_coordinates[i]])
        max_y = max([j[0] for j in cc_coordinates[i]])

        try:
            # Check if the aspect ratio of the connected component is between 1.5 and 5.0
            if 1.5 <= (max_x - min_x)/(max_y - min_y) <= 5.0:
                # Add the coordinates of the corners of the connected component to the dictionary
                return min_x, max_x, min_y, max_y
        except ZeroDivisionError:
            pass

# This function filters the image using a Gaussian filter
def computeGaussianAveraging11x11RepeatBorder(pixel_array, image_width, image_height, min_x, max_x, min_y, max_y):
    filtered_image = createInitializedGreyscalePixelArray(image_width, image_height)

    # BorderBoundaryPadding
    for i in range(0, 5):
        pixel_array.insert(0, pixel_array[1])
        pixel_array.append(pixel_array[len(pixel_array)-1])

    for i in range(0, len(pixel_array)):
        for j in range(0, 5):
            pixel_array[i].insert(0, pixel_array[i][0])
            pixel_array[i].append(pixel_array[i][-1])

    # Filter the image
    for i in range(5, image_height+5):
        for j in range(5, image_width+5):
            if min_x <= j <= max_x and min_y <= i <= max_y:
                window = pixel_array[i-5][j-5:j+6] + pixel_array[i-4][j-5:j+6] + pixel_array[i-3][j-5:j+6] + pixel_array[i-2][j-5:j+6] + pixel_array[i-1][j-5:j+6] + pixel_array[i][j-5:j+6] + pixel_array[i+1][j-5:j+6] + pixel_array[i+2][j-5:j+6] + pixel_array[i+3][j-5:j+6] + pixel_array[i+4][j-5:j+6] + pixel_array[i+5][j-5:j+6]
                filtered_image[i-5][j-5] = abs(sum(window)/121.0)
            else:
                filtered_image[i-5][j-5] = pixel_array[i][j]

    return filtered_image

# END STUDENT IMPLEMENTATION

# This is our code skeleton that performs the license plate detection.
# Feel free to try it on your own images of cars, but keep in mind that with our algorithm developed in this lecture,
# we won't detect arbitrary or difficult to detect license plates!
def main():
    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    input_filename = "numberplate6.png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images2")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(input_filename.replace(".png", "_output2.png"))
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate result   s in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')

    # STUDENT IMPLEMENTATION
    # Conversion to greyscale, and contrast stretch
    greyscale_pixel_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    contrast_stretched_pixel_array1 = scaleTo0And255AndQuantize(greyscale_pixel_array, image_width, image_height)

    # Filtering to detect high contrast regions, and contrast stretch
    filtered_image1 = computeStandardDeviationImage5x5(contrast_stretched_pixel_array1, image_width, image_height)
    contrast_stretched_pixel_array2 = scaleTo0And255AndQuantize(filtered_image1, image_width, image_height)

    # Thresholding for segmentation
    thresholded_image = computeThresholdGE(contrast_stretched_pixel_array2, 150, image_width, image_height)

    # Dilation and erosion
    closed_image = thresholded_image
    n = 4
    # Dilation
    for i in range(0, n):
        closed_image = computeDilation8Nbh3x3FlatSE(closed_image, image_width, image_height)

    # Erosion
    for i in range(0, n):
        closed_image = computeErosion8Nbh3x3FlatSE(closed_image, image_width, image_height)

    # Labeling
    cc_img, cc_sizes, cc_coordinates = computeConnectedComponentLabeling(closed_image, image_width, image_height)

    # Compute a bounding box
    bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y = computeBoundingBox(cc_coordinates, cc_sizes)

    # Filtering to blur the license plate
    filtered_image2 = greyscale_pixel_array
    for i in range(0, 10):
        filtered_image2 = computeGaussianAveraging11x11RepeatBorder(filtered_image2, image_width, image_height, bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y)

    px_array = filtered_image2
    # END STUDENT IMPLEMENTATION

    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1, edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()
