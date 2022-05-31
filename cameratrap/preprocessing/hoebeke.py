import math
import numpy as np

from scipy import ndimage as ndi
from skimage import measure
from skimage.filters import roberts
from PIL import Image, ImageChops, ImageFilter

def standard_box(box, standard_length, standard_height, image_length, image_height):
   """Returns the boundaries of a box with specified length en height, centered around a
   given smaller box. The obtained bigger box lies within the image.

   Parameters
   ----------
   box :
      original smaller box
   standard_length :
      length of the standard box
   standard_height :
      height of the standard box
   image_length :
      length of the image
   image_height :
      height of the image

   Returns
   -------
   box_standard :
      boundaries of the standard box (left, top, right, bottom)

   """


   box = np.asarray(box)
   length_box = box[2]-box[0]
   height_box = box[3]-box[1]


   if box[0]-(standard_length-length_box)/2 < 0: #left outside the image
        box[0] = 0
        box[2] = standard_length
   elif box[2]+(standard_length-length_box)/2 > image_length: #right outside the image
        box[0] = image_length-standard_length-1
        box[2] = image_length-1
   else:
        box[0] = box[0]-math.floor((standard_length-length_box)/2)
        box[2] = box[2]+math.ceil((standard_length-length_box)/2)


   if box[1]-(standard_height-height_box)/2 < 0: #top outside the image
        box[1] = 0
        box[3] = standard_height
   elif box[3]+(standard_height-height_box)/2 > image_height: #bottom outside the image
        box[1] = image_height-standard_height-1
        box[3] = image_height-1
   else:
        box[1] = box[1]-math.floor((standard_height-height_box)/2)
        box[3] = box[3]+math.ceil((standard_height-height_box)/2)

   box_standard = tuple(box)

   return box_standard


def black_border(image):

   """Returns the boundaries of the image without the black borders at
   the top and bottom of the image, containing metadata.

   Parameters
   ----------
   image: PIL.Image

   Returns
   -------
   image_box:
      Boundaries of the image without borders (left, top, right, bottom)
   """

   image = image.convert('L')
   image_length = image.size[0]
   image_height = image.size[1]

   column = np.asarray(image)[:,0]
   top = next((i for i, x in enumerate(column) if x))
   bottom = (image_height-1) - next((i for i, x in enumerate(column[::-1]) if x))
   image_box = tuple([0, top, image_length-1, bottom])

   return image_box


def size_box(box):

   """Calculates the length and the height of a box.

   Arguments:
   box: boundaries of the box (left, top, right, bottom)

   Return:
   length_box: length of the box
   height_box: height of the box

   """
   length_box = box[2]-box[0]
   height_box = box[3]-box[1]

   return [length_box, height_box]


def divide_box(box, length_standard_box, height_standard_box, image_length, image_height):
   """Divides a box that is bigger than the standard box, into standard boxes.
   The obtained standard boxes lie within the image.

   Parameters
   ----------
   box:
      original box
   standard_length:
      length of the standard box
   standard_heighth:
      height of the standard box
   image_length:
      length of the image
   image_height:
      height of the image

   Returns
   -------
   boxes: list
      list containing the boundaries of the boxes (left, top, right, bottom)
   """
   length_box = size_box(box)[0]
   height_box = size_box(box)[1]

   #Option 1: box is longer than the standard box, but not higher
   if length_box > length_standard_box and height_box < height_standard_box:

       left_box = np.asarray(box)
       left_box[2] = left_box[0]+length_standard_box
       left_box = standard_box(left_box, length_standard_box, height_standard_box, image_length, image_height)

       right_box = np.asarray(box)
       right_box[0] = right_box[2]-length_standard_box
       right_box = standard_box(right_box,length_standard_box,height_standard_box, image_length, image_height)

       boxes = [tuple(left_box), tuple(right_box)]

   #Option 2: box is higher than the standard box, but not longer
   elif length_box < length_standard_box and height_box > height_standard_box:

       top_box = np.asarray(box)
       top_box[3] = top_box[1]+height_standard_box
       top_box = standard_box(top_box, length_standard_box, height_standard_box, image_length, image_height)

       bottom_box = np.asarray(box)
       bottom_box[1] = bottom_box[3]-height_standard_box
       bottom_box = standard_box(bottom_box,length_standard_box,height_standard_box, image_length, image_height)

       boxes = [tuple(top_box), tuple(bottom_box)]

   #Option 3: box is higher and longer than the standard box
   else:

       topleft_box = np.asarray(box)
       topleft_box[2] = topleft_box[0]+length_standard_box
       topleft_box[3] = topleft_box[1]+height_standard_box

       topright_box = np.asarray(box)
       topright_box[0] = topright_box[2]-length_standard_box
       topright_box[3] = topright_box[1]+height_standard_box

       bottomleft_box = np.asarray(box)
       bottomleft_box[2] = bottomleft_box[0]+length_standard_box
       bottomleft_box[1] = bottomleft_box[3]-height_standard_box

       bottomright_box = np.asarray(box)
       bottomright_box[0] = bottomright_box[2]-length_standard_box
       bottomright_box[1] = bottomright_box[3]-height_standard_box

       boxes = [tuple(topleft_box), tuple(topright_box), tuple(bottomleft_box), tuple(bottomright_box)]

   return boxes


def box_center(box):

   """Returns the coördinates of the center of a box.

   Arguments:
   box: box (left, top, right, bottom)

   Return:
   center: (left, top)

   """
   box = np.array(box)
   left = math.ceil(np.mean([box[0], box[2]]))
   top = math.ceil(np.mean([box[1], box[3]]))
   center = tuple((left,top))

   return center


def extract_boxes(image, background_image):
   """Extract the regions of interest on an image

   Parameters
   ----------
   image : PIL.Image
      Image to extract boxes for.
   background_image : PIL.Image
      Background image used for extraction (e.g. median of sequence).

   Notes
   -----
   Adjusted original code of L. Hoebeke as little as possible
   (not clear what the different options refer to; requires a unit test data set to do so)
   """
   image_length, image_height = image.size[0], image.size[1]

   # Parameters for box extraction
   ratio = 0.5  # hardcoded, as used as such in remaining model definition
   min_pixel_diff = int(500*ratio**2)
   max_pixel_diff = image_length*image_height*0.6
   min_pixel_object = int(6000*ratio**2)
   length_standard_box = 960*ratio  # size standard box
   height_standard_box = 540*ratio  # size standard box

   # Parameters for binary closing
   struct = np.ones((20, 20)).astype(int)
   iter_closing = 5 #number of iterations

   # Difference with background
   diff = ImageChops.difference(background_image, image).convert('L')

   # MinFilter
   filterf = diff.filter(ImageFilter.MinFilter(size=9))

   # Number of pixels that are different
   pixels_filter = np.count_nonzero(np.asarray(filterf))
   box_filter = filterf.getbbox()

   # No (significant) difference with background
   if not isinstance(box_filter, tuple) or pixels_filter < min_pixel_diff:
      return [()], [()]  # box, box_small

   # Too much difference with background
   elif pixels_filter > max_pixel_diff:
      box_object_list = divide_box(image.getbbox(), length_standard_box,
                                    height_standard_box, image_length, image_height)
      return box_object_list, [()]  # box, box_small

   else:
      length_box_filter = size_box(box_filter)[0]
      height_box_filter = size_box(box_filter)[1]

      # Box after filtering is smaller than standard box
      if length_box_filter < length_standard_box and height_box_filter < height_standard_box:
         box = standard_box(box_filter,
                              length_standard_box, height_standard_box,
                              image_length, image_height)
         return [box], [box_filter]

      # Box after filtering is larger than standard box
      else:
         # Edge detection
         edge = roberts(filterf)

         # MinFilter after edge detection
         edge = (edge != 0).astype(int)
         edge = Image.fromarray(edge.astype('uint8')).filter(ImageFilter.MinFilter(size=3))
         edge = Image.fromarray(np.asarray(edge).astype('uint8')).filter(ImageFilter.MinFilter(size=3))

         # Binary closing
         closing = ndi.binary_closing(edge, structure=struct, iterations=iter_closing,
                                       output=None, origin=0)

         # Connected component labeling
         # NOTE: converted the neighbours=8 to connectivity=2
         connect = measure.label(closing, connectivity=2, background=0, return_num=True)
         counts = np.bincount(connect[0].flatten())

         # Box after connected component labeling
         box = Image.fromarray(closing.astype('uint8')).getbbox()

         if not isinstance(box, tuple):  # empty box
               return [()], [()] # box, box_small

         else:
               length_box = size_box(box)[0]
               height_box = size_box(box)[1]

               # Box after connected component labeling is larger than standard box
               if length_box > length_standard_box or height_box > height_standard_box:

                  # Boxes around objects
                  box_object_list = []
                  box_object_list_small = []

                  for a in range(1, (connect[1])+1):

                     if counts[a] > min_pixel_object:

                           box_object = Image.fromarray((connect[0]==a).astype('uint8')).getbbox()
                           box_object_list_small.append(box_object)

                           length_box_object = size_box(box_object)[0]
                           height_box_object = size_box(box_object)[1]

                           # Box around object bigger than standard box
                           if length_box_object > length_standard_box or height_box_object > height_standard_box:

                              boxes = divide_box(box_object, length_standard_box,
                                                height_standard_box, image_length, image_height)
                              box_object_list += boxes

                           # Box around object is smaller than standard box
                           else:
                              box_object = standard_box(box_object,length_standard_box,height_standard_box, image_length, image_height)
                              box_object_list += [box_object]  # .append(box_object)

                  if not box_object_list:
                     box_object_list = [()]
                     box_object_list_small = [()]
                  return box_object_list, box_object_list_small

               # Box after connected component labeling is smaller than standard box
               else:
                  box_standard = standard_box(box, length_standard_box,
                                             height_standard_box,
                                             image_length, image_height)
                  return [box_standard], [box] # box, box_small