import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def nadji_sum(image, skok):
    tacke=list()
    for i in range(200, 400):
        for j in range(200, 400):
            if image[i][j] - image[i-1][j] >= skok or image[i][j] - image[i+1][j] >= skok or image[i][j] - image[i][j-1] >= skok or image[i][j] - image[i][j+1] >= skok:
                tacke.append((i, j))
    return tacke

def ukloni_sum(image, tacke):
    for t in tacke:
        filter = (image[t[0]][t[1]] + image[t[0]-1][t[1]] + image[t[0]-1][t[1]-1] + image[t[0]][t[1]-1] + image[t[0]+1][t[1]-1] + image[t[0]+1][t[1]] + image[t[0]+1][t[1]+1] + image[t[0]][t[1]+1] + image[t[0]-1][t[1]+1]) / 9
        image[t[0]][t[1]] = filter
    return image

original_image = cv.imread('input.png', 0)

cv.imshow("Pocetna slika", original_image)

ft = np.fft.fft2(original_image) 
ft_shift = np.fft.fftshift(ft)

magnitude_spectrum = 20 * np.log(np.abs(ft_shift))

plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnituda spektra pre uklanjanja suma:')
plt.savefig('fft_mag.png')
plt.show()

possible_states = nadji_sum(magnitude_spectrum, 67) 
ft_shift = ukloni_sum(ft_shift, possible_states)

magnitude_spectrum = 20 * np.log(np.abs(ft_shift))

plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnituda spektra nakon uklanjanja suma:')
plt.savefig('fft_mag_filtered.png')
plt.show()

in_ft_shift = np.fft.ifftshift(ft_shift)

img_processed = np.fft.ifft2(in_ft_shift).real

cv.imshow("Finalna slika: ", img_processed.astype(np.uint8))
cv.imwrite('output.png', img_processed)

cv.waitKey(0)
cv.destroyAllWindows()
