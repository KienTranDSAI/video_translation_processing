from landmarks_detection import *
import cv2
import numpy as np
from color_transfer import *

import sys
sys.path.append("/home/kientran/Code/Work/Overlayed video/RestoreFormerPlusPlus")
sys.path.insert(1,"/home/kientran/Code/Work/Overlayed video/RestoreFormerPlusPlus/app.py")
from app import *

cap = cv2.VideoCapture("/home/kientran/Code/Work/Overlayed video/Blend_lipsync/Blended different video/My Video/output (5).mp4")
baseCap = cv2.VideoCapture("/home/kientran/Code/Work/Overlayed video/Blend_lipsync/Blended different video/My Video/video_test2.mp4")

widthBase = int(baseCap.get(3))
heightBase = int(baseCap.get(4))
widthInsert = int(cap.get(3))
heightInsert = int(cap.get(4))

##Write video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/home/kientran/Code/Work/Overlayed video/Blend_lipsync/Blended different video/myVideo.mp4', fourcc, 30.0, (int(baseCap.get(3)), int(baseCap.get(4))))

##Create landmark detector
landmarks_detector = create_landmarks_dectection_model()
enhance_image = True
make_color_transfer = False
while cap.isOpened() and baseCap.isOpened():

    insertRet, insertFrame = cap.read()
    baseRet, baseFrame = baseCap.read()

    if enhance_image:
    ##Using RestoreFormer model to enhance image
        tempFrame, savepath = video_inference(insertFrame,'RestoreFormer', 'no', 2) # 'no': no align, 's': scale number
        tempFrame = cv2.cvtColor(tempFrame, cv2.COLOR_RGB2BGR)      #Convert to BGR (default format of cv2)
        insertFrame = cv2.resize(tempFrame, insertFrame.shape[:2]) ### Return to original shape

    if insertRet and baseRet:
        basePolyPoint = []
        insertPolyPoint = []

        cv2.imshow('InsertFrame', insertFrame)

        ##Determine mouth points by landmark from mediapipe
        
        # mouth_points = [212,186,92,165,167,164,393,391,322,410,432,273,335,406,313,18,83,182,106,43]
        mouth_points = [192, 207,206,165,167,164,393,391,426,427,416,364,394,395,369,396,175,171,140,170,169,135]
        
        ## Determine landmark for two frame
        insertLandmarks = get_landmarks(landmarks_detector, insertFrame)
        baseLandmarks = get_landmarks(landmarks_detector, baseFrame)

        print(insertLandmarks)
        ##Create mask of mouth position
        insertMask = np.zeros(insertFrame.shape[:2], dtype='uint8')
        baseMask = np.zeros(baseFrame.shape[:2], dtype='uint8')
        if insertLandmarks != [] and baseLandmarks != []:
            for point in mouth_points:
                inser_x,insert_y = insertLandmarks[0][point].x, insertLandmarks[0][point].y
                inser_x = int(inser_x*widthInsert)
                insert_y = int(insert_y*heightInsert)
                insertPolyPoint.append([inser_x,insert_y])

                

                base_x, base_y = baseLandmarks[0][point].x,baseLandmarks[0][point].y
                base_x = int(base_x*widthBase)
                base_y = int(base_y * heightBase)
                basePolyPoint.append([base_x, base_y])
            
            ##Create mask 
            insertPoint = np.array(insertPolyPoint,dtype=np.int32)
            cv2.fillPoly(insertMask, pts=[insertPoint], color=(255,255,255))
            basePoint = np.array(basePolyPoint, dtype=np.int32)
            cv2.fillPoly(baseMask, pts = [basePoint], color = (255,255,255))


            ##Warp image for mouth positions matching
            baseMask = cv2.bitwise_not(baseMask)
            matrix, mask = cv2.findHomography(np.array([insertPolyPoint]), np.array([basePolyPoint]), cv2.RANSAC, 5.0)

            warppedFrame = cv2.warpPerspective(insertFrame, matrix, (widthBase, heightBase))
            warppedMask = cv2.warpPerspective(insertMask, matrix, (widthBase, heightBase))
            finalMask = cv2.bitwise_and(warppedMask, cv2.bitwise_not(baseMask))
            finalMask[finalMask > 1] = 255
            finalMask[finalMask <=1] = 0
            baseMask = cv2.bitwise_not(finalMask)

            #Blur mask
            myMask = cv2.blur(finalMask, (9,9))
            myMask = myMask/255
            myMask = np.stack((myMask,myMask,myMask), -1)

            maskedBase = (cv2.bitwise_and(baseFrame, baseFrame, mask = baseMask)/255).astype(np.float32) 
            maskedInsert = (cv2.bitwise_and(warppedFrame, warppedFrame, mask = finalMask)/255).astype(np.float32) 
            ## Transform frame from int8 to float for color transfer 
            floatBaseFrame = (baseFrame/255).astype(np.float32)
            finalMask = (finalMask/255).astype(np.float32)
            warppedFrame = (warppedFrame/255).astype(np.float32)
            

            result = cv2.add(maskedBase, maskedInsert)
            result = floatBaseFrame * (1- myMask) + myMask * warppedFrame

            if make_color_transfer:
                ## Find rectangular box for color transfer
                min_x_base = min([i[0] for i in basePolyPoint]) 
                min_y_base = min([i[1] for i in basePolyPoint]) 
                max_x_base = max([i[0] for i in basePolyPoint]) 
                max_y_base = max([i[1] for i in basePolyPoint]) 

                #Get mouth box of two frame for color transfer
                a = (result[min_y_base:max_y_base,min_x_base:max_x_base,:]/255).astype(np.float32)
                b = (baseFrame[min_y_base:max_y_base,min_x_base:max_x_base,:]/255).astype(np.float32)
                # ans = color_transfer_mkl(a,b)
            
                ans = color_hist_match(a,b)
                ans = linear_color_transfer(ans,b)
            

                result[min_y_base:max_y_base,min_x_base:max_x_base,:] = ans
            #Blend with pure frame to remove rectangle border       
            result = floatBaseFrame * (1- myMask) + myMask * result

            ##Convert frame into integer to write out
            normalized_array = (result - np.min(result))/(np.max(result) - np.min(result)) # this set the range from 0 till 1
            img_array = (normalized_array * 255).astype(np.uint8)
            out.write(img_array)
        else:
            out.write(baseFrame)
        # cv2.imshow("Result",result)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release() 
out.release()
cv2.destroyAllWindows() 
