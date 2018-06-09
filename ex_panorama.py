def find_homo(im1,im2):
    sift=cv2.xfeatures2d.SIFT_create()

    kp1,des1=sift.detectAndCompute(im1,None)
    kp2,des2=sift.detectAndCompute(im2,None)
    
    bf=cv2.BFMatcher()
    matches=bf.knnMatch(des1,des2,k=2)

    mat1=[]
    mat2=[]
    
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            mat1.append(kp1[m.queryIdx])
            mat2.append(kp2[m.trainIdx])
    
    pts1=np.float32([m.pt for m in mat1])
    pts2=np.float32([m.pt for m in mat2])

    H, status = cv2.findHomography(pts2, pts1, cv2.RANSAC, 3.0)
    return H

def make_homo_img(h, img1, img2):
    
    homo=np.ones((3,1))
    homo[0,0]=img2.shape[1]/2
    homo[1,0]=img2.shape[0]/2
    new_homo=np.dot(h,homo)

    for i in range(new_homo.shape[0]):
        new_homo[i,0]=new_homo[i,0]/new_homo[2,0]
        
    dx=int(new_homo[0,0]-homo[0,0])


    if dx<0:

        dx=np.absolute(dx)
        y=img1.shape[0]
        x=img1.shape[1]+dx
        new_img=np.zeros((y, x, 3), np.uint8)
        new_h=np.array([[1,0,dx], [0,1,0], [0,0,1]], np.float32)
        h=np.dot(new_h,h)
        
        trans_img1=cv2.warpPerspective(img2, h, (img2.shape[1], img2.shape[0]))
        
        for i in range(3):
            new_img[:,dx:,i]=img1[:,:,i]
        
        none_zero=((new_img[:,:img2.shape[1],0]+new_img[:,:img2.shape[1],1]+new_img[:,:img2.shape[1],2])>0)
        
        for i in range(3):
            new_img[:,:img2.shape[1],i] = trans_img1[:,:img2.shape[1],i]*(1-none_zero) + new_img[:,:img2.shape[1],i]*none_zero


    else:
        trans_img1=cv2.warpPerspective(img2, h, (img2.shape[1]+dx, img2.shape[0]))

        y=img1.shape[0]
        x=img2.shape[1]+dx
        
        new_img=np.zeros((y, x, 3),np.uint8)

        for i in range(3):
            new_img[:,:,i]=trans_img1[:,:,i]
        none_zero=((img1[:,:,0]+img1[:,:,1]+img1[:,:,2])>0)
        for i in range(3):
            new_img[:,:img1.shape[1],i]=img1[:,:,i]*none_zero+ new_img[:,:img1.shape[1],i]*(1-none_zero)

    return new_img, trans_img1


import cv2
import numpy as np


img1=cv2.imread('pano_img1.jpg')
img2=cv2.imread('pano_img2.jpg')
img3=cv2.imread('pano_img3.jpg')
#img3=cv2.imread('pano_img4.jpg')
#img4=cv2.imread('pano_img4.jpg')

H=find_homo(img1,img2)
result_img_1,trans_=make_homo_img(H,img1,img2)
H1=find_homo(result_img_1,img3)
result_img_2,trans_=make_homo_img(H1,result_img_1,img3)
#H1=find_homo(result_img_2,img4)
#result_img_3,trans_=make_homo_img(H1,result_img_2,img4)

cv2.imshow("panorama", result_img_1)
cv2.imshow("trans image", trans_)
cv2.imshow("panorama1", result_img_2)
#cv2.imshow("panorama2", result_img_3)
cv2.waitKey()
cv2.destroyAllWindows()

