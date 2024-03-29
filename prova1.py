import cv2
import numpy as np
from matplotlib import pyplot as plt

def media(imagem):

    img_final = np.zeros(imagem.shape, imagem.dtype)
    for j in range(1, imagem.shape[0] - 1):
        for i in range(1, imagem.shape[1] - 1):
            valor = (int(imagem[j - 1, i - 1]) + int(imagem[j - 1, i]) + int(imagem[j - 1, i + 1]) + \
                         int(imagem[j, i - 1]) + int(imagem[j, i]) + int(imagem[j, i + 1]) \
                        + int(imagem[j + 1, i - 1]) + int(imagem[j + 1, i]) + int(imagem[j + 1, i + 1]))/9
            if valor > 255:
                valor = 255
            if valor < 0:
                valor = 0
            img_final[j, i] = valor

    return img_final

def transformacaoLinear(imagem, k1, k2, l1, l2):
    img = imagem
    for k in range(imagem.shape[2]):
        for y in range(imagem.shape[0]):
            for x in range(imagem.shape[1]):
                if imagem[y,x,k] < l1:
                    valor = k1

                elif imagem[y,x,k] >= l1 and imagem[y,x,k] < l2:
                    valor = int((((k2-k1)/(l2-l1))*(imagem[y,x,k]-l1)+k1)-100)

                elif imagem[y,x,k] >= l2:
                    valor = k2 - 100

                if valor > 255:
                    valor = 255
                if valor < 0:
                    valor = 0
                    
                img[y,x,k] = valor        

    return img
        
def questao2(image):
        print("Processando... Aguarde\n")        
        k1, k2, l1, l2 = 0, 255, 100, 220
        imgT = transformacaoLinear(image, k1, k2, l1, l2)
        cv2.imshow("Imagem T. Linear", imgT)
        plt.hist(imgT.ravel(), 256, [0, 256])
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def questao1(img):
    print("Aguarde, as imagens estão sendo gravadas")
    #Operador Canny
    can = cv2.imread("prova-01.ppm")
    img_canny = cv2.Canny(can,100,200)
    
    #Operador Laplaciano
    laplacian = cv2.Laplacian(img,cv2.CV_8U)

    #Operador Gaussiano
    gaussianBlur = cv2.GaussianBlur(img,(5,5),0)

    #Operador Média
    img_media = media(img)

    #Operador Sobel 
    img = cv2.imread("prova-01.ppm")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    sobel = cv2.bitwise_or(sobelX, sobelY)
    resultado = np.vstack([
                np.hstack([img, sobelX]),
                np.hstack([sobelY, sobel])
            ])

    #Roberts
    kernelRobertx = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
    kernelRoberty = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])
    robertx = cv2.filter2D(img, -1, kernelRobertx)
    roberty = cv2.filter2D(img, -1, kernelRoberty)
    img_robert = robertx + roberty
    

    cv2.imwrite("Canny.png", img_canny)
    
    cv2.imwrite("Roberts.png", img_robert)
    cv2.imwrite("Sobel.png", sobel)
    cv2.imwrite("Gaussiano.png", gaussianBlur)
    cv2.imwrite("Laplacian.png", laplacian)
    cv2.imwrite("Media.png", img_media)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Prewitt
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    img_prewittx = cv2.filter2D(img, -1, kernelx)
    img_prewitty = cv2.filter2D(img, -1, kernely)
    img_prewitt = img_prewittx + img_prewitty
    cv2.imwrite("Prewitt.png", img_prewitt)
    print("Imagens gravadas com sucesso!")
    
while(1):
    print("Prova | Informe o número da questão:\n1 - Questão 1\n2 - Questão 2\n0 - Sair")
    entrada = (input(""))

    if entrada == '1':
        img = cv2.imread("prova-01.ppm",0)
        questao1(img)
            
    if entrada == '2':         
        img = cv2.imread("prova-01.ppm")
        img = img[::6, ::6]
        questao2(img)
        
    if entrada == '0':
        break;

    
