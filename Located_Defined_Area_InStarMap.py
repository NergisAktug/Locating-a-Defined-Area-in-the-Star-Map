import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

#formul: GlobalOrtalamaAltSinir<=PencereOrtalama<=GlobalOrtalamaUstSinir: k0*Mglobal<=Mpencere<=k1*Mglobal
#formul: GlobalStandartSapmaAltSinir<=PencereStandartSapma=GlobalStandartSapmaUstSinir: k2*STDglobal<=STDpencere<=k3*STDglobal
def local_histogram_statistics(StarMap,SmallArea,window_size):
    StarMap=StarMap.astype(float)
    SmallArea=SmallArea.astype(float)
    
    globalMean=np.mean(SmallArea)
    StdGlobal=np.std(SmallArea)
       
    M,N=StarMap.shape[:2] #StarMapin (y,x) degerini aldim.SmallArea boyutundaki cerceve StarMap matrixinde gezinecek
    Indexs=[]
    for row in range(M):
        if M-row>=window_size:
            upper=max(row,row-window_size)
            lower=min(M,row+window_size)
        for column in range(N):
            if N-column>=window_size:
                left=max(column,column-window_size)
                right=min(N,column+window_size)
                windowsPixels=StarMap[upper:lower,left:right]   
                window_mean=np.mean(windowsPixels)
                window_std=np.std(windowsPixels)
                if window_mean==globalMean and window_std==StdGlobal:
                    Indexs.append([upper,left])
                    Indexs.append([upper,right-1])
                    Indexs.append([lower-1,left])
                    Indexs.append([lower-1,right-1])
                    
    Indexs=np.uint32(Indexs)#satirlar float olamaz
    return Indexs
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))   

StarMap=cv2.imread("./brief/StarMap.png",0)   #StarMap dosyasinı OpenCv library ile aldim   
Small_Area=cv2.imread("./brief/Small_area.png",0)#Small_Area dosyasinı OpenCv library ile aldim
Small_area_rotated=cv2.imread("./brief/Small_area_rotated.png",0)

M,N=Small_Area.shape[:2]
X,Y=Small_area_rotated.shape[:2]

Located_Small_Area=local_histogram_statistics(StarMap,Small_Area,M)
print("Located_Small_Area[Row,Column]:",Located_Small_Area[0:4])
#Small_area_rotated=cv2.rotate(Small_area_rotated,cv2.ROTATE_180)
Located_Small_area_rotated=local_histogram_statistics(StarMap,Small_area_rotated,X)
angle=1
while len(Located_Small_area_rotated)==0 and angle!=360:
    Small_area_rotated=cv2.imread("./brief/Small_area_rotated.png",0)
    Small_area_rotated=rotate_bound(Small_area_rotated,angle)
    angle+=1
    Located_Small_area_rotated=local_histogram_statistics(StarMap,Small_area_rotated,X)


print("Located_Small_area_rotated[Row,Column]:",Located_Small_area_rotated[0:4])




