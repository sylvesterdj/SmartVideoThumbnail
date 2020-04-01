# Program To Read video 
# and Extract Frames 
import os
import numpy as np
import cv2 
from skimage.measure import compare_ssim 
import face_recognition

if __name__ == '__main__': 
    
    # Base directory
    baseDirectory = "C:\\MediaFiles"
    # movies = ['Petta','Bigil','Bigil','Frozen2','SCOOB','Spider Man','THOR','Toy Story']
    movies = ['Petta']
    isWriteAllFrames=True

    for movie in movies:
        inputFileName = baseDirectory + '\\' + movie + '.MP4'
        movieOutputPath = baseDirectory + '\\' + movie
        allFrameOutputPath = movieOutputPath + '\\allFrames' 

        # classifiers = ['haarcascade_upperbody.xml','haarcascade_frontalface_alt.xml','haarcascade_frontalface_alt2.xml','haarcascade_frontalface_default.xml','haarcascade_profileface.xml']
        classifiers = ['haarcascade_frontalface_alt.xml']
        # faceClassifiers = ['haarcascade_eye.xml','haarcascade_smile.xml']

        # create directory
        if(not os.path.exists(movieOutputPath)):
            os.mkdir(movieOutputPath)

        if(not os.path.exists(allFrameOutputPath)):
            os.mkdir(allFrameOutputPath)

        # inputPath to video file 
        vidObj = cv2.VideoCapture(inputFileName) 

        # Used as counter variable
        count = 0

        # checks whether frames were extracted 
        success = 1

        # store previous image for comparison
        previousBestImage = None
        previousBestImageCount = 0
        previousBestImageBlur = 0

        # initialize our lists of extracted facial embeddings and
        # corresponding people names
        knownEncodings = []
        knownIds = {}

        while success: 

            # vidObj object calls read 
            # function extract frames 
            success, image = vidObj.read()

            print(count)

            # if(count < 3580) :
            #     count += 1
            #     continue
            # elif (count > 3584):
            #     break

            if(image is not None):

                if(isWriteAllFrames):
                    # Saves the frames with frame-count 
                    cv2.imwrite(allFrameOutputPath+"\\frame%d.jpg" % count, image)

                # initialize image per classifier to show highlighed area without conflict
                faceImage = None
                # print(movieOutputPath, count, classifiers)
                # minFaceSizes = [(50,50), (100,100), (200,200), (300,300)]
                minFaceSizes = [(300,300)]
                maxFaceSize = (700,700)

                color = [(0,255,0,),(0,0,255,),(0,100,100,),(100,0,100,),(100,100,0,),(150,150,150,)]
                for classifier in classifiers:

                    classifierOutputPath = movieOutputPath + '\\' + classifier

                    if(not os.path.exists(classifierOutputPath)):
                        os.mkdir(classifierOutputPath)

                    # Load the cascade
                    face_cascade = cv2.CascadeClassifier(classifier)
                    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

                    faceImage = image

                    # Convert to grayscale
                    grey = cv2.cvtColor(faceImage, cv2.COLOR_BGR2GRAY)
                    
                    for minFaceSize in minFaceSizes:
                        faceOutputPath = classifierOutputPath + "\\"+ str(minFaceSize)

                        if(not os.path.exists(faceOutputPath)):
                            os.mkdir(faceOutputPath)

                        # Detect the faces
                        # faces = face_cascade.detectMultiScale(grey, 1.1, 4, minSize=minFaceSize, maxSize=maxFaceSize)
                        faces = face_cascade.detectMultiScale(grey, 1.1, 4, minSize=minFaceSize, maxSize=maxFaceSize)

                        if (len(faces) > 0):
                            # draw rectangle on identified areas
                            # for (x, y, w, h) in faces:
                            #     # cv2.rectangle(faceImage, (x, y), (x+w, y+h), (255,0,0,), 2)

                            #     roi_grey = grey[y:y+h, x:x+w]
                            #     roi_color = faceImage[y:y+h, x:x+w]

                            #     i = 0

                            #     for faceClassifier in faceClassifiers:
                            #         # Load the cascade
                            #         facial_feature_cascade = cv2.CascadeClassifier(faceClassifier)

                            #         face_features = facial_feature_cascade.detectMultiScale(roi_grey, 1.1, 4, minSize=minFaceSize, maxSize=maxFaceSize)

                            #         if (len(face_features) > 0):
                            #             # draw rectangle on identified areas
                            #             for (x, y, w, h) in face_features:
                            #                 cv2.rectangle(roi_color, (x, y), (x+w, y+h),color[i], 2)

                            #         i = i+1
                            # Saves the frames with frame-count 
                            cv2.imwrite(faceOutputPath +"\\frame%d.jpg" % count, faceImage)

                            # fs.selectBestFace(faceOutputPath)
                            bestFaceOutputPath = faceOutputPath + "\\bestFaces"

                            if(not os.path.exists(bestFaceOutputPath)):
                                os.mkdir(bestFaceOutputPath)
                                                    
                            if(previousBestImage is not None):
                                # Convert to grayscale
                                previousBestImageGrey = cv2.cvtColor(previousBestImage, cv2.COLOR_BGR2GRAY)

                                # Convert to grayscale
                                grey = cv2.cvtColor( faceImage, cv2.COLOR_BGR2GRAY)

                                # compute the Structural Similarity Index (SSIM) between the two
                                # images, ensuring that the difference image is returned
                                (score, diff) = compare_ssim(previousBestImageGrey, grey, full=True)
                                # print("SSIM: {} old frame %d new frame %d".format(score) %(previousBestImageCount, count))

                                # compute the Laplacian of the image and then return the focus
                                # measure, which is simply the variance of the Laplacian
                                fm = cv2.Laplacian(grey, cv2.CV_64F).var()

                                if(score > 0.6):
                                    # print("Current image blur %d previous image blur %d" %(fm, previousBestImageBlur))

                                    if(previousBestImageBlur < fm):
                                        print("choosing current frame %d with previous blur %d current blur %d" %(count, previousBestImageBlur, fm))                            
                                        # if current image is better than previous image
                                        previousBestImage = faceImage
                                        previousBestImageCount = count
                                        previousBestImageBlur = fm
                                else:
                                    # Write previous good image before processing new image that's not similar to previous image
                                    cv2.imwrite(bestFaceOutputPath+"\\frame%d.jpg" % previousBestImageCount, previousBestImage)

                                    # if current image is better than previous image
                                    previousBestImage =  faceImage
                                    previousBestImageCount = count
                                    previousBestImageBlur = fm
                            else:
                                #first image
                                previousBestImage = faceImage
                                previousBestImageCount = count

                            # cf.identifyUniqueCharacters(faceOutputPath)
                            minArea = 2000
                            uniqueFaceOutputPath = faceOutputPath + '\\uniqueFaces'
                            if(not os.path.exists(uniqueFaceOutputPath)):
                                os.mkdir(uniqueFaceOutputPath)

                            boxes = face_recognition.face_locations(faceImage)
                            if(len(boxes) > 0):
                                # print("boxes",boxes)
                                encodings = face_recognition.face_encodings(faceImage, boxes)
                                # print(encodings)
                                i = 0
                                for encoding in encodings:
                                    (top, right, bottom, left) = boxes[i]
                                    # print(i)
                                    matches = face_recognition.compare_faces(knownEncodings, encoding)

                                    face_distances = face_recognition.face_distance(knownEncodings, encoding)
                                    face_distance = 0
                                    if(len(face_distances) > 0):
                                        # print('distance',face_distances)
                                        best_match_index = np.argmin(face_distances)
                                        face_distance = face_distances[best_match_index]

                                    if (True not in matches or face_distance > 0.5) :
                                        # write face to file
                                        
                                        # print(boxes[i])
                                        # cv2.rectangle(faceImage, (top, right), (bottom, left), (0, 255, 0), 2)
                                        face_image = faceImage[top:bottom, left:right]
                                        (w,h,d) = face_image.shape
                                        # print('face shape',w,h)

                                        area = w * h
                                        #  consider only larger faces
                                        # print(area)

                                        if (area > minArea):
                                            # print('Face considered')
                                            print("New face in frame",count)
                                            # add encoding to known list
                                            knownEncodings.append(encoding)

                                            # print(len(knownEncodings))
                                            # add as new id and frame count
                                            faceId = len(knownEncodings)-1
                                            knownIds[faceId] = 1
                                            # print(knownIds)
                                        

                                            # print("[INFO] Object found. Saving locally."+roi_color)
                                            # cv2.imwrite(outputPath+str(count)+'-'+str(i) + '_faces.jpg', roi_color)
                                            cv2.imwrite(uniqueFaceOutputPath+"\\face-frame-%d-%d.jpg" % (count,faceId), face_image)

                                            framesByFace = uniqueFaceOutputPath + "\\" + str(faceId)
                                            if(not os.path.exists(framesByFace)):
                                                os.mkdir(framesByFace)

                                            cv2.imwrite(framesByFace+"\\frame-%d-%d.jpg" % (count,faceId), faceImage)
                                        else:
                                            # print('Faces filtered')

                                            # print("[INFO] Object found. Saving locally."+roi_color)
                                            # cv2.imwrite(outputPath+str(count)+'-'+str(i) + '_faces.jpg', roi_color)
                                            filteredFaces = uniqueFaceOutputPath + "\\filteredFaces"
                                            if(not os.path.exists(filteredFaces)):
                                                os.mkdir(filteredFaces)

                                            cv2.imwrite(filteredFaces+"\\face-frame-%d.jpg" % (count), face_image)
                                    else:
                                        face_image = faceImage[top:bottom, left:right]
                                        (w,h,d) = face_image.shape
                                        # print('face shape',w,h)

                                        area = w * h
                                        #  consider only larger faces
                                        # print(area)
                                        
                                        if (area > minArea):
                                            # print('face id',best_match_index)
                                            if matches[best_match_index]:
                                                # faceId = knownEncodings[best_match_index]
                                                knownIds[best_match_index] = knownIds[best_match_index] + 1
                                                framesByFace = uniqueFaceOutputPath + "\\" + str(best_match_index)
                                                cv2.imwrite(framesByFace+"\\frame-%d-%d.jpg" % (count,best_match_index), faceImage)
                                        else:
                                            filteredFaces = uniqueFaceOutputPath + "\\filteredFaces"
                                            if(not os.path.exists(filteredFaces)):
                                                os.mkdir(filteredFaces)

                                            cv2.imwrite(filteredFaces+"\\face-frame-%d.jpg" % (count), face_image)

                                    # print(knownIds)
                                    i = i+1

                            print('Unique faces',len(knownEncodings))
                            print(knownIds)

                count += 1