import os
import cv2
import dlib
from imutils import face_utils
import numpy as np
import torch
from torchvision import transforms
from model import gaze_network

from head_pose import HeadPoseEstimator

trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
#     ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)

#     ## further optimize
#     if iterate:
#         ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

#     return rvec, tvec
def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    # Ensure correct data types and dimensions
    face_model = np.array(face_model, dtype=np.float32).reshape(-1, 1, 3)  # (n,1,3)
    landmarks = np.array(landmarks, dtype=np.float32).reshape(-1, 1, 2)    # (n,1,2)
    
    # Verify minimum 4 points for EPnP
    if len(face_model) < 4:
        raise ValueError("Need at least 4 points for EPnP algorithm")
    
    # First estimation
    ret, rvec, tvec = cv2.solvePnP(
        objectPoints=face_model,
        imagePoints=landmarks,
        cameraMatrix=camera,
        distCoeffs=distortion,
        flags=cv2.SOLVEPNP_EPNP
    )
    
    # Refinement
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(
            objectPoints=face_model,
            imagePoints=landmarks,
            cameraMatrix=camera,
            distCoeffs=distortion,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
    return rvec, tvec

def draw_gaze(image_in, pitchyaw, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = np.min([h, w]) / 2.0
    pos = (int(w / 2.0), int(h / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)

    return image_out

def normalizeData_face(img, face_model, landmarks, hr, ht, cam):
    ## normalized camera parameters
    focal_norm = 960  # focal length of normalized camera
    distance_norm = 600  # normalized distance between eye and camera
    roiSize = (224, 224)  # size of cropped eye image

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht  # rotate and translate the face model
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    # get the face center
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))

    ## ---------- normalize image ----------
    distance = np.linalg.norm(face_center)  # actual distance between eye and original camera

    z_scale = distance_norm / distance
    cam_norm = np.array([  # camera intrinsic parameters of the virtual camera
        [focal_norm, 0, roiSize[0] / 2],
        [0, focal_norm, roiSize[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([  # scaling matrix
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T  # rotation matrix R

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # transformation matrix

    img_warped = cv2.warpPerspective(img, W, roiSize)  # warp the input image

    # head pose after normalization
    hR_norm = np.dot(R, hR)  # head pose rotation matrix in normalized space
    hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

    # normalize the facial landmarks
    num_point = landmarks.shape[0]
    landmarks_warped = cv2.perspectiveTransform(landmarks, W)
    landmarks_warped = landmarks_warped.reshape(num_point, 2)

    return img_warped, landmarks_warped
def get_gaze_direction(image, cam_matrix, cam_distortion):
    """Process image and return gaze direction"""
    # Face detection
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb_image, 1)
    if not faces:
        return None, None, None

    # Landmark detection
    shape = predictor(image, faces[0])
    landmarks = face_utils.shape_to_np(shape)
    
    # Head pose estimation
    # face_model = np.loadtxt('face_model.txt')[[20,23,26,29,15,19]]
    # hr, ht = estimateHeadPose(landmarks, face_model, cam_matrix, cam_distortion)
    face_model = np.loadtxt('face_model.txt')[landmark_use, :]  # Ensure this loads 6+ points
    landmarks_sub = landmarks[[36,39,42,45,31,35], :].astype(np.float32)

    hr, ht = estimateHeadPose(
        landmarks=landmarks_sub,
        face_model=face_model,
        camera=camera_matrix,
        distortion=camera_distortion
    )
    print("kkkk")
    
    # Data normalization
    img_normalized, _ = normalize_face_data(image, face_model, landmarks, hr, ht, cam_matrix)
    
    # Gaze prediction
    input_tensor = trans(img_normalized[:, :, [2,1,0]]).unsqueeze(0)
    with torch.no_grad():
        gaze_pred = model(input_tensor)[0].cpu().numpy()
    
    return gaze_pred, hr, ht

def create_screen_heatmap(gaze_vector, hr, ht, cam_matrix):
    """Generate screen heatmap with gaze projection"""
    # Capture screen
    screenshot = pyautogui.screenshot()
    screen_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    h, w = screen_img.shape[:2]
    
    # Convert gaze angles to 3D vector
    pitch, yaw = gaze_vector
    gaze_3d = np.array([
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
        np.cos(pitch) * np.cos(yaw)
    ])
    
    # Transform using head pose
    R = cv2.Rodrigues(hr)[0]
    gaze_3d = R @ gaze_3d
    
    # Project to screen coordinates (simplified perspective projection)
    screen_x = int(w/2 * (1 + gaze_3d[0] * 1.5))  # Scale factor for better visibility
    screen_y = int(h/2 * (1 - gaze_3d[1] * 1.5))
    
    # Generate heatmap
    heatmap = np.zeros((h, w), dtype=np.float32)
    radius = int(min(h, w) * 0.15)
    cv2.circle(heatmap, (screen_x, screen_y), radius, 1, -1)
    heatmap = cv2.GaussianBlur(heatmap, (0,0), radius//3)
    
    # Create overlay
    heatmap_colored = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(screen_img, 0.7, heatmap_colored, 0.3, 0)
    cv2.circle(blended, (screen_x, screen_y), 10, (0,255,255), -1)
    
    return blended
if __name__ == '__main__':
    img_file_name = './example/input/test.webp'
    print('load input face image: ', img_file_name)
    image = cv2.imread(img_file_name)

    predictor = dlib.shape_predictor('./modules/shape_predictor_68_face_landmarks.dat')
    # face_detector = dlib.cnn_face_detection_model_v1('./modules/mmod_human_face_detector.dat')
    face_detector = dlib.get_frontal_face_detector()  ## this face detector is not very powerful
    detected_faces = face_detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1) ## convert BGR image to RGB for dlib
    if len(detected_faces) == 0:
        print('warning: no detected face')
        exit(0)
    print('detected one face')
    shape = predictor(image, detected_faces[0]) ## only use the first detected face (assume that each input image only contains one face)
    shape = face_utils.shape_to_np(shape)
    landmarks = []
    for (x, y) in shape:
        landmarks.append((x, y))
    landmarks = np.asarray(landmarks)

    # load camera information
    cam_file_name = './example/input/cam00.xml'  # this is camera calibration information file obtained with OpenCV
    if not os.path.isfile(cam_file_name):
        print('no camera calibration file is found.')
        exit(0)
    fs = cv2.FileStorage(cam_file_name, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode('Camera_Matrix').mat() # camera calibration information is used for data normalization
    camera_distortion = fs.getNode('Distortion_Coefficients').mat()

    print('estimate head pose')
    # load face model
    face_model_load = np.loadtxt('face_model.txt')  # Generic face model with 3D facial landmarks
    landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
    face_model = face_model_load[landmark_use, :]
    # estimate the head pose,
    ## the complex way to get head pose information, eos library is required,  probably more accurrated
    # landmarks = landmarks.reshape(-1, 2)
    # head_pose_estimator = HeadPoseEstimator()
    # hr, ht, o_l, o_r, _ = head_pose_estimator(image, landmarks, camera_matrix[cam_id])
    ## the easy way to get head pose information, fast and simple
    facePts = face_model.reshape(6, 1, 3)
    landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :]
    landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
    landmarks_sub = landmarks_sub.reshape(6, 1, 2)  # input to solvePnP requires such shape
    hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion)

    # data normalization method
    print('data normalization, i.e. crop the face image')
    img_normalized, landmarks_normalized = normalizeData_face(image, face_model, landmarks_sub, hr, ht, camera_matrix)

    print('load gaze estimator')
    model = gaze_network()
    model.load_state_dict(torch.load('./ckpt/epoch_24_ckpt.pth.tar', map_location='cpu')['model_state'])
    model.eval()

    
    fs = cv2.FileStorage('./example/input/cam00.xml', cv2.FILE_STORAGE_READ)
    cam_matrix = fs.getNode('Camera_Matrix').mat()

    cam_distortion = fs.getNode('Distortion_Coefficients').mat()

    
    # Process input image
    input_img = cv2.imread('./example/input/test.webp')
    gaze_vector, hr, ht = get_gaze_direction(input_img, cam_matrix, cam_distortion)
    
    if gaze_vector is not None:
        # Generate and save heatmap
        result = create_screen_heatmap(gaze_vector, hr, ht, cam_matrix)
        cv2.imwrite('gaze_heatmap_screenshot.jpg', result)
        print("Successfully saved gaze heatmap screenshot!")
    else:
        print("No face detected in input image")