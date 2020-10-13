import pandas as pd
import numpy as np
import cv2
import argparse

parser= argparse.ArgumentParser()
parser.add_argument('-i', '--frame', required=True, help= 'Input the image path, if using video stream enter "None"')
parser.add_argument('-c', '--conf',default=0.7, type= float , help='Input confidence value')

arguments_of_image= vars(parser.parse_args())

print('Lets Load the model')
face_mod= cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')


if arguments_of_image['frame']== 'None':
    
    cam= cv2.VideoCapture(0)
    while(True):
        grab, frame= cam.read()
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        print('Computing Detections')
        (h,w)= frame.shape[:2]
        face_mod.setInput(blob)
        predictions= face_mod.forward()
        
        for i in range(0, predictions.shape[2]):
            confidence= predictions[0, 0 , i ,2 ]
            if confidence > arguments_of_image['conf']:
                
                box= predictions[0,0,i,3:7]*np.array([w,h,w,h])
                (startX, startY, endX, endY)= box.astype('int')
                
                text= 'Hooman = {:.2f}%'.format(confidence*100)
                y= startY-10 if startY-10>10 else startY+10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,255),2)
                cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255 , 0), 2)
        cv2.imshow('Output', frame)
        if cv2.waitKey(1)== 13:
            break
    cam.release()
    cv2.destroyAllWindows()

else:
    frame= cv2.imread(arguments_of_image['frame'])
    (h,w)= frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
    print('Computing Detections')
    face_mod.setInput(blob)
    predictions= face_mod.forward()
        
    for i in range(0, predictions.shape[2]):
        confidence= predictions[0 , 0 , i , 2]

        if confidence > arguments_of_image['conf']:
            box= predictions[0,0,i,3:7]*np.array([w,h,w,h])

            (startX, startY, endX, endY)= box.astype('int')
            
            text= 'Hooman = {:.2f}%'.format(confidence*100)
            y= startY-10 if startY-10>10 else startY+10

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255),2)

            cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255 , 0), 2)
    cv2.imshow('Output', frame)
    print("cv2.imshow()")
    cv2.waitKey(0)
    cv2.destroyAllWindows()