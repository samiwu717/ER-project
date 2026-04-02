import cv2
for i in [0, 1]:
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ok, f = cap.read()
    cv2.imshow(f"Camera {i}", f)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    cap.release()
    