import cv2 as cv
from cv2 import aruco
import numpy as np
import glob

MARKER_SIZE = 2
MARKER_SEPARATION = 0.4
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
arucoParams = aruco.DetectorParameters_create()
board = aruco.GridBoard_create(5, 7, MARKER_SIZE, MARKER_SEPARATION, aruco_dict)

def calibrate(files : str):
    counter, corners_list, id_list = [], [], []
    images = glob.glob(files)

    h,w = 0,0

    for (i, img) in enumerate(images):
        image = cv.imread(str(img))
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
        if i == 0:
            h,w = image.shape[:2]
            corners_list = corners
            id_list = ids
        else:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list,ids))
        counter.append(len(ids))

    counter = np.array(counter)
    _, mtx, dist, _, _ = aruco.calibrateCameraAruco(corners_list,  id_list, counter, board, img_gray.shape, None, None)

    ncm, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    return (mtx, dist, ncm, roi)

def video(cap, camera_matrix, dist_coeffs, new_camera_matrix, roi):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = np.array(frame)
        frame_remapped_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_remapped_gray, aruco_dict, parameters=arucoParams)
        aruco.refineDetectedMarkers(frame_remapped_gray, board, corners, ids, rejectedImgPoints)

        im_with_aruco_board = frame

        if type(ids) is np.ndarray and ids.any() != None:
            im_with_aruco_board = aruco.drawDetectedMarkers(frame, corners, ids, (0,255,0))
            rvec = ()
            tvec = ()
            retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs, rvec, tvec)
            if retval != 0:
                im_with_aruco_board = cv.drawFrameAxes(im_with_aruco_board, camera_matrix, dist_coeffs, rvec, tvec, 5)

        undst = cv.undistort(im_with_aruco_board, camera_matrix, dist, None, new_camera_matrix)

        width = int(undst.shape[1] * 0.6)
        height = int(undst.shape[0] * 0.6)
        dim = (width, height)
        undst = cv.resize(undst, dim, interpolation = cv.INTER_AREA)

        cv.imshow("video", undst)
        if cv.waitKey(2) & 0xFF == ord('q'):
            break

    return

if __name__ == '__main__':
    mtx, dist, nmc, roi = calibrate('Aruco/*.jpg')
    cap = cv.VideoCapture("Aruco/Aruco_board.mp4")
    video(cap, mtx, dist, nmc, roi)
    cap.release()
    cv.destroyAllWindows()