import numpy as np
import cv2 as cv

slika1 = cv.imread('1.jpg')
slika2 = cv.imread('2.jpg')
slika3 = cv.imread('3.jpg')

def napraviPanoramu():
    img = NapraviPanoramuOdDveSlike(slika2, slika3)
    img = NapraviPanoramuOdDveSlike(slika1, img)
    return img

def NapraviPanoramuOdDveSlike(imgL, imgR):
    detector = cv.SIFT_create() # inicijalizacija
   
   # Trazimo karakteristicne tacke, kao rezultat dobijamo niz katakteristicnih tacaka i
   # vektor koji predstavlja sliku
    kp1, des1 = detector.detectAndCompute(imgR, None) 
    kp2, des2 = detector.detectAndCompute(imgL, None)

    #spaja tacke
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50) 

    # Pozivamo filtriranje tacaka:
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Rezultat je niz tipa: [[<DMatch>, <DMatch>], [<DMatch>, <DMatch>], [<DMatch>, <DMatch>]...],
    # odnosno dobijamo niz objekta, sa kojima na zalost sledeca funkcija ne ume da radi, pa moramo
    # da ih transformisemo u niz tipa: [[float32, float32], [float32, float32], [float32, float32]...]

   # Iako prethodni korak vraca 50 najslicnihij, moguce je da se desi da je max slicnost tek 15%,
   # ovako, odbacujemo sve parove iz niza cija je slicnost manja od 70%
   # DMatch.distance - mera slicnosti;

    # prag slicnosti
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    #odbacuje lazne parove
    # Ako imamo barem 30 tacaka koje se poklapaju na obe slike, onda nastavljamo sa radom:
    if len(good) > 30:

    # Reshape: oblikuje niz, u ovom slucaju dobijamo sledeci oblik:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), 30))
        return None
    

    # Spajamo 2 slike u 1 (sa crnom popunom za mesta koja se ne nalaze ni na jednoj od dve spojene slike);

    # spaja slike
    width = imgL.shape[1] + imgR.shape[1]
    height = imgL.shape[0] + int(imgR.shape[0] / 2)
    outimg = cv.warpPerspective(imgR, M, (width, height))
    outimg[0:imgL.shape[0], 0:imgL.shape[1]] = imgL

    # Uklanja redove i vrste piksela koji su od pocetka do kraja crni, dakle visak:
    #sece crno sa strane
    outimg = trim(outimg)
    return outimg

def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    if not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])
    if not np.sum(frame[:, -1]):
        return trim(frame[:, :-2])
    return frame

panorama = napraviPanoramu()

cv.imshow("Panorama", panorama)
cv.imwrite("output.jpg", panorama)
cv.waitKey(0)
