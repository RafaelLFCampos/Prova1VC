import cv2
import numpy as np
from matplotlib import pyplot as plt

def matriz(imagem):
    for y in range(imagem.shape[0]):
        for x in range(imagem.shape[1]):
            imagem[y,x] = 255
            
def transformacaoLinear(imagem, k1, k2, l1, l2):
    img_escura = np.ones(imagem.shape, imagem.dtype)
    matriz(img_escura)
    img_media = np.zeros(imagem.shape, imagem.dtype)
    img_clara = np.zeros(imagem.shape, imagem.dtype)
    
    
    for y in range(imagem.shape[0]):
        for x in range(imagem.shape[1]):
            if imagem[y,x] < l1:
                img_escura[y,x] = imagem[y,x]

            elif imagem[y,x] >= l1 and imagem[y,x] < l2:
                img_media[y,x] = imagem[y,x]

            elif imagem[y,x] >= l2:
                img_clara[y,x] = imagem[y,x]      

    cv2.imshow("Imagem Escura", img_escura)
    cv2.imshow("Imagem Media", img_media)
    cv2.imshow("Imagem Clara", img_clara)
    cv2.imshow("Imagem Original", imagem)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    plt.hist(img_escura.ravel(), 256, [0, 254])
    plt.show()
    
    plt.hist(img_media.ravel(), 256, [1, 256])
    plt.show()
    
    plt.hist(img_clara.ravel(), 256, [1, 256])
    plt.show()
    cv2.destroyAllWindows()
        
def b(image):
    while(1):
                
        new_image = np.zeros(image.shape, image.dtype)
        alpha = 1.0 # Simple contrast control
        beta = -100    # Simple brightness control    
        #alpha = float(input('* Enter the alpha value [1.0-3.0]: '))
        #beta = int(input('* Enter the beta value [0-100]: '))
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(image.shape[2]):
                    new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)

        cv2.imshow("Imagem nova", new_image)
        cv2.waitKey(0)
        

img = cv2.imread("prova-01.ppm", 0)
img = img[::6, ::6]
plt.hist(img.ravel(), 256, [0, 256])
plt.show()
cv2.imshow("OriginalL", img)
cv2.waitKey(0)
k1, k2, l1, l2 = 0, 255, 25, 200
transformacaoLinear(img, k1, k2, l1, l2)
