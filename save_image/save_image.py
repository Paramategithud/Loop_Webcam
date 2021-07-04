#pip install opencv_python
#pip install tensorflow
#pip install keras
#pip install Pillow
import cv2
import tensorflow
import keras 
import numpy as np
from PIL import Image
value=int(input("Enter mask=1 , non_mask=2 : ")) 
webcam = cv2.VideoCapture(0)
success, image_bgr = webcam.read()
face_cascade = "haarcascade_frontalface_default.xml"
count = 1
if value == 1:
    while True:
        success,image_bgr = webcam.read()
        image_org = image_bgr.copy()
        image_bw = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        face_classifier = cv2.CascadeClassifier(face_cascade)
        faces = face_classifier.detectMultiScale(image_bw)
        print(f'There are {len(faces)} faces found.')
        for face in faces:
            x, y, w, h = face
            cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite(f'mask/mask_{count}.jpg',image_org[y:y+h,x:x+w])
            count += 1 
        cv2.imshow("Faces found",image_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

elif value == 2:
    while True:
        success,image_bgr = webcam.read()
        image_org = image_bgr.copy()
        image_bw = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        face_classifier = cv2.CascadeClassifier(face_cascade)
        faces = face_classifier.detectMultiScale(image_bw)
        print(f'There are {len(faces)} faces found.')
        for face in faces:
            x, y, w, h = face
            cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite(f'nonmask/non_mask_{count}.jpg',image_org[y:y+h,x:x+w])
            count += 1
        cv2.imshow("Faces found",image_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:
    face_classifier = cv2.CascadeClassifier(face_cascade)
    # Disable scientific notation for clarity
    #np.set_printoptions(suppress=True)
    model = tensorflow.keras.models.load_model('keras_model.h5')
    size = (224, 224)
    while True:
        success,image_bgr = webcam.read()
        image_org = image_bgr.copy()
        image_bw = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        faces = face_classifier.detectMultiScale(image_bw)
        #print(f'There are {len(faces)} faces found.')
        for face in faces:
            x, y, w, h = face
            cface_rgb = Image.fromarray(image_rgb[y:y+h,x:x+w])
            # Create the array of the right shape to feed into the keras model
            # The 'length' or number of images you can put into the array is
            # determined by the first position in the shape tuple, in this case 1.
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            # Replace this with the path to your image
            #image = Image.open('test_photo.jpg')
            image = cface_rgb
            #resize the image to a 224x224 with the same strategy as in TM2:
            #resizing the image to be at least 224x224 and then cropping from the center
            image = ImageOps.fit(image, size, Image.ANTIALIAS)
            #turn the image into a numpy array
            image_array = np.asarray(image)
            # display the resized image
            #image.show()
            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            # Load the image into the array
            data[0] = normalized_image_array
            # run the inference
            prediction = model.predict(data)
            print(prediction)
            if(prediction[0][0] > prediction[0][1]):
                #FONT_HERSHEY_SCRIPT_SIMPLEX
                cv2.putText(image_bgr,'Mask',(x,y-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                cv2.rectangle(image_bgr,(x, y),(x+w, y+h),(0,255,0),2)
            else:
                cv2.putText(image_bgr,'Non_Mask',(x,y-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                cv2.rectangle(image_bgr,(x, y),(x+w, y+h),(0,0,255),2)   
        cv2.imshow("Mask_detection",image_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
