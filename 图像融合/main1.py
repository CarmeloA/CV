from demo01 import *

k = cv.imread('knife.jpg')
bg = cv.imread('bg3.JPEG')

knife,k_thresh,x,y,w,h,r = knife_pretreatment(k,170)
print(knife.shape)
k_thresh_inv = knife_pretreatment_inv(k,170)
k_thresh_inv = k_thresh_inv[y:y + h, x:x + w]

k_thresh = cv.cvtColor(k_thresh,cv.COLOR_GRAY2BGR).astype(np.uint8)
new_k = knife * ((k_thresh/255).astype(np.uint8))


bg_roi = bg[300:300+h,300:300+w]
k_thresh_inv = cv.cvtColor(k_thresh_inv,cv.COLOR_GRAY2BGR).astype(np.uint8)
bg_roi = bg_roi*((k_thresh_inv/255).astype(np.uint8))


bg_roi = bg_roi+new_k
bg[300:300+h,300:300+w] = bg_roi

cv.imshow('res',bg)

cv.waitKey(0)