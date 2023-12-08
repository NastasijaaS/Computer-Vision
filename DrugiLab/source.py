import numpy as np
import cv2 as cv

image = cv.imread('coins.png')
cv.imshow("Original: ", image)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#svi pikseli ispod praga postavljeni na belu boju, a svi pikseli iznad praga su crni. 
_, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV) 

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))

# smanjuje sum i spaja regione
closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

coins, _ = cv.findContours(closing, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

mask = np.zeros_like(closing)
main_mask = np.zeros_like(closing)
main_mask.fill(0)

for coin in coins:

    # Izdvajamo samo element za koji smo zainteresovani:
    mask.fill(0)
    cv.drawContours(mask, [coin], -1, 255, -1)
    
    # Racunamo prosecnu RGB vrednost za svaki objekat (prosek za svaki element RGB strukture);
    avg_color = cv.mean(image, mask=mask)[:3]

    # Prosecna boja srebrnog novcica:
    bg_color = [128, 128, 128]

    # Racunamo razliku vektora 
    color_diff = np.linalg.norm(np.array(avg_color) - np.array(bg_color))

    if color_diff > 45.0:
        cv.drawContours(main_mask, [coin], -1, 255, -1)

cv.imshow('Maska: ', main_mask)
cv.imwrite("coin_mask.png", main_mask)

result = cv.bitwise_and(image, image, mask= main_mask)

cv.imshow('Maskirani novcic: ', result)
cv.imwrite("result.png", result)

cv.waitKey(0)
cv.destroyAllWindows()
