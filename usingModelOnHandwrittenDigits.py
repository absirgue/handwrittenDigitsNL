'''
Runs model on handritten digits of our creation (use Paint or Scan digits and resize to 28*28pixels to have fun with it!)
Save your handwritten digits in a digits folder as the root of the project with file name 'digit{nb of the image starting at 1}'

Author: @asirgue
Version: 2.0 edited on 19/11 before uploading to Github
'''
import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('handwritten.model')

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0] # Only caring about shapes not colours
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}") # Returning the value of the neuron with the highest activation (the most likely to be the actual digit)
        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()
    except:
        print('Error') # If resolution is not the right definition and our model can't handle it
    
    finally:
        image_number += 1 # Incrementing image_number to move on to our next handwritten digit file
