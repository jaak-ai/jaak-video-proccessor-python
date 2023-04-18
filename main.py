import cv2
import dlib
import base64
import requests
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

cap = cv2.VideoCapture(int(config['camera']['source']))
detector = dlib.get_frontal_face_detector()


def main():

    while True:
        _, frame = read_video_frames()
        faces = detect_faces(frame)

        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            roi = crop_image(frame, x1, y1, x2, y2)
            b64 = image_to_base64(roi)

            url = config['predictor']['url']
            body = {'image': b64}
            headers = {"Authorization": config['predictor']['authorization']}
            x = requests.post(url, json=body, headers=headers)
            response = x.json()
            print(x.status_code)

            if response.get("user"):
                cv2.putText(
                    frame,
                    response.get("user").get("fullName"),
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )



            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # cv2.imshow("roi", roi_color)

        cv2.imshow("Video Proccessor", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break


def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    bimage = base64.b64encode(buffer)
    return bimage.decode("utf-8")


def crop_image(frame, x1, y1, x2, y2):
    return frame[y1:y2, x1:x2]


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return detector(gray)


def read_video_frames():
    return cap.read()


if __name__ == "__main__":
    main()
