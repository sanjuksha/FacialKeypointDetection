import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#matplotlib inline
import cv2
# load in color image for face detection
image = cv2.imread('images/image12.jpeg')

# switch red and blue color channels 
# --> by default OpenCV assumes BLUE comes first, not RED as in many images
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plot the image
fig = plt.figure(figsize=(9,9))
plt.imshow(image)


# load in a haar cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# run the detector
# the output here is an array of detections; the corners of each detection box
# if necessary, modify these parameters until you successfully identify every face in a given image
faces = face_cascade.detectMultiScale(image, 1.2, 2)

# make a copy of the original image to plot detections on
image_with_detections = image.copy()

# loop over the detected faces, mark the image where each face is found
for (x,y,w,h) in faces:
    # draw a rectangle around each detected face
    # you may also need to change the width of the rectangle drawn depending on image resolution
    cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3) 

fig = plt.figure(figsize=(9,9))

plt.imshow(image_with_detections)
import torch
from models import Net

net = Net()

## TODO: load the best saved model parameters (by your path name)
## You'll need to un-comment the line below and add the correct name for *your* saved model
net.load_state_dict(torch.load('saved_models/keypoints_model_10.pt'))

## print out your net and prepare it for testing (uncomment the line below)
net.eval()

image_copy = np.copy(image)
from torch.autograd import Variable
# loop over the detected faces from your haar cascade
for (x,y,w,h) in faces:
    
    # Select the region of interest that is the face in the image 
    roi = image_copy[y:y+h, x:x+w]
    #print(roi.shape)
    ## TODO: Convert the face region from RGB to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    
    ## TODO: Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
    norm_roi = gray_roi/255.0
    ## TODO: Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
    roi_1 = cv2.resize(norm_roi,(224,224))
    roi_rescale = cv2.resize(norm_roi,(224,224))
    roi_1 = roi_1.reshape(roi_1.shape[0], roi_1.shape[1], 1)
    
    ## TODO: Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
    roi_1 = roi_1.reshape(1,224,224)
    #plt.imshow(roi_2, cmap='gray')
    #print(roi_1.shape)
    ## TODO: Make facial keypoint predictions using your loaded, trained network 
    roi_1 = torch.from_numpy(roi_1)
    #print(roi_tensor.shape)
    # Make facial keypoint predictions using loaded, trained network
    roi_1= roi_1.unsqueeze_(0)
    roi_1 = roi_1.type(torch.FloatTensor)
    roi_1 = Variable(roi_1)
    output_pts = net(roi_1)
    # Display each detected face and the corresponding keypoints
    output_pts = output_pts.view(output_pts.size()[0], 68, -1)
    output_pts = output_pts.data.numpy()
    #output_pts = output_pts.numpy()
    output_pts = output_pts*75.0+100
    #print(output_pts.shape)

    fig = plt.figure()
    fig.add_subplot(1,len(faces),1)
    plt.imshow(np.squeeze(roi_rescale), cmap='gray')
    plt.scatter(output_pts[:,:,0],output_pts[:,:,1] ,s=20, marker='.', c='m')
    plt.show()
    ## TODO: Display each detected face and the corresponding keypoints    

plt.show()