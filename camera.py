import cv2


class Camera:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise Exception("Cannot open camera.")
        test_frame = self.read()
        if test_frame is None:
            print("Camera read failed.")
            self.release()
            return None

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Cannot read frame.")
            return None
        return frame

    def release(self):
        self.cap.release()

    def show(self, frame):
        cv2.imshow("USB Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True


if __name__ == '__main__':
    cap = Camera(1)
    while True:
        frame = cap.read()
        cap.show(frame)
    