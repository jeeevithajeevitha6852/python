import os
import argparse
import cv2  # Ensure OpenCV is installed: pip install opencv-python

# Argument Parser
parser = argparse.ArgumentParser(description="Detect age and gender from an image or webcam")
parser.add_argument('--image', help='Path to input image file')
args = parser.parse_args()

# Model Files
faceProto = "deploy.prototxt"
faceModel = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Load Networks
faceNet = cv2.dnn.readNetFromCaffe(faceProto, faceModel)
ageNet = cv2.dnn.readNetFromCaffe(ageProto, ageModel)
genderNet = cv2.dnn.readNetFromCaffe(genderProto, genderModel)

# Preprocessing Constants
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
padding = 20

# Face Detection Function
def highlightFace(net, frame, threshold=0.7):
    frameCopy = frame.copy()
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frameCopy, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameCopy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frameCopy, faceBoxes

# Main Logic
def detectAgeGender(frame):
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")
        return resultImg

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
                     max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]

        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]

        label = f"{gender}, {age}"
        cv2.putText(resultImg, label, (faceBox[0], faceBox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

    return resultImg

# Run on Image or Webcam
if args.image and os.path.isfile(args.image):
    img = cv2.imread(args.image)
    output = detectAgeGender(img)
    cv2.imshow("Age and Gender Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Opening webcam. Press ESC to quit.")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        output = detectAgeGender(frame)
        cv2.imshow("Age and Gender Detection", output)
        if cv2.waitKey(1) == 27:  # ESC key to exit
            break
    cap.release()
    cv2.destroyAllWindows()
