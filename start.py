import cv2
import dlib
import numpy as np
from imutils import face_utils

# Load the pre-trained face detector and facial landmark predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Download this from dlib

# Load the sunglasses image with transparency (RGBA)
sunglasses = cv2.imread('sunglasses.png', -1)

# 3D model points of facial landmarks
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),         # Chin
    (-225.0, 170.0, -135.0),      # Left eye left corner
    (225.0, 170.0, -135.0),       # Right eye right corner
    (-150.0, -150.0, -125.0),     # Left mouth corner
    (150.0, -150.0, -125.0)       # Right mouth corner
], dtype=np.float32)

# Camera internals
size = (640, 480)
focal_length = size[1]
center = (size[1] / 2, size[0] / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype=np.float32
)

dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

def add_3d_sunglasses(frame, landmarks, rotation_vector, translation_vector):
    # 2D landmark points for placing sunglasses
    left_eye = (landmarks[36][0], landmarks[36][1])
    right_eye = (landmarks[45][0], landmarks[45][1])

    # Calculate sunglasses width and height based on 3D rotation
    sunglasses_width = int(np.linalg.norm(np.array(left_eye) - np.array(right_eye)) * 2)
    sunglasses_height = int(sunglasses_width * sunglasses.shape[0] / sunglasses.shape[1])

    # Resize sunglasses based on head pose
    resized_sunglasses = cv2.resize(sunglasses, (sunglasses_width, sunglasses_height))

    # Get the region of interest on the frame
    x_offset = left_eye[0] - sunglasses_width // 4
    y_offset = left_eye[1] - sunglasses_height // 2
    y1, y2 = y_offset, y_offset + sunglasses_height
    x1, x2 = x_offset, x_offset + sunglasses_width

    # Get the alpha channel from the sunglasses image to use as a mask
    alpha_sunglasses = resized_sunglasses[:, :, 3] / 255.0
    alpha_frame = 1.0 - alpha_sunglasses

    # Overlay the sunglasses on the frame based on head pose
    for c in range(0, 3):
        frame[y1:y2, x1:x2, c] = (alpha_sunglasses * resized_sunglasses[:, :, c] +
                                  alpha_frame * frame[y1:y2, x1:x2, c])

    return frame

# Start the webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # 2D image points
        image_points = np.array([
            (landmarks[33][0], landmarks[33][1]),  # Nose tip
            (landmarks[8][0], landmarks[8][1]),    # Chin
            (landmarks[36][0], landmarks[36][1]),  # Left eye left corner
            (landmarks[45][0], landmarks[45][1]),  # Right eye right corner
            (landmarks[48][0], landmarks[48][1]),  # Left mouth corner
            (landmarks[54][0], landmarks[54][1])   # Right mouth corner
        ], dtype=np.float32)

        # Solve PnP for head pose estimation
        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

        # Add 3D sunglasses to the face
        frame = add_3d_sunglasses(frame, landmarks, rotation_vector, translation_vector)

    # Display the result
    cv2.imshow("3D Sunglasses Filter", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
