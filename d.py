import os
import cv2
import dlib
from imutils import face_utils
import numpy as np
import torch
from torchvision import transforms
from model import gaze_network
import mss
from PIL import Image


from head_pose import HeadPoseEstimator

trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this for production security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)

    ## further optimize
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

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

    return img_warped, landmarks_warped, R

def draw_3d_gaze(image, eye_pos, gaze_vec, length=200):
    # Ensure inputs are NumPy arrays and flatten them
    eye_pos = np.array(eye_pos).flatten().astype(int)
    gaze_vec = np.array(gaze_vec).flatten()
    
    # Calculate end point (using only x,y components of gaze vector)
    end_point = eye_pos[:2] + length * gaze_vec[:2]  # Use first 2 dimensions
    end_point = end_point.astype(int)
    
    # Convert to tuples with native Python integers
    eye_pos = tuple(eye_pos.tolist())
    end_point = tuple(end_point.tolist())
    
    # Draw arrow
    cv2.arrowedLine(image, 
                    eye_pos,
                    end_point,
                    (0, 255, 0), 2)       


if __name__ == '__main__':
    img_file_name = './example/input/cam00.jpg'
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
    img_normalized, landmarks_normalized ,R= normalizeData_face(image, face_model, landmarks_sub, hr, ht, camera_matrix)

    print('load gaze estimator')
    model = gaze_network()
    # model.cuda() # comment this line out if you are not using GPU
    pre_trained_model_path = './ckpt/epoch_24_ckpt.pth.tar'
    if not os.path.isfile(pre_trained_model_path):
        print('the pre-trained gaze estimation model does not exist.')
        exit(0)
    else:
        print('load the pre-trained model: ', pre_trained_model_path)
    ckpt = torch.load(pre_trained_model_path,map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model_state'], strict=True)  # load the pre-trained model
    model.eval()  # change it to the evaluation mode
    input_var = img_normalized[:, :, [2, 1, 0]]  # from BGR to RGB
    input_var = trans(input_var)
    input_var = torch.autograd.Variable(input_var.float().to('cpu'))
    input_var = input_var.view(1, input_var.size(0), input_var.size(1), input_var.size(2))  # the input must be 4-dimension
    pred_gaze = model(input_var)  # get the output gaze direction, this is 2D output as pitch and raw rotation
    pred_gaze = pred_gaze[0] # here we assume there is only one face inside the image, then the first one is the prediction
    pred_gaze_np = pred_gaze.cpu().data.numpy()  # convert the pytorch tensor to numpy array

    print('prepare the output')
    # draw the facial landmarks
    landmarks_normalized = landmarks_normalized.astype(int) # landmarks after data normalization
    for (x, y) in landmarks_normalized:
        cv2.circle(img_normalized, (x, y), 5, (0, 255, 0), -1)
    face_patch_gaze = draw_gaze(img_normalized, pred_gaze_np)  # draw gaze direction on the normalized face image
    output_path = 'example/output/results_gaze.jpg'
    print('save output image to: ', output_path)
    cv2.imwrite(output_path, face_patch_gaze)
        # Convert gaze angles to vector in normalized space
    
    
    
    
   # --------------------------
# MODIFIED HEAD POSE AND GAZE CALCULATION
# --------------------------

# After getting pred_gaze_np in main()
    pitch, yaw = pred_gaze_np

    gaze_vector_normalized = np.array([
        np.sin(yaw) * np.cos(pitch),  # Removed negative sign
        np.sin(pitch),                # Removed negative sign
        np.cos(yaw) * np.cos(pitch)
    ])
    gaze_vector_normalized /= np.linalg.norm(gaze_vector_normalized)

    # 2. Transform gaze vector to ORIGINAL CAMERA SPACE
    _, _, R = normalizeData_face(image, face_model, landmarks_sub, hr, ht, camera_matrix)
    gaze_vector_camera = R.T @ gaze_vector_normalized.reshape(3, 1)
    # Rotate gaze vector to original camera coordinates
    gaze_vector_camera = R.T @ gaze_vector_normalized.reshape(3, 1)

    # 3. Calculate 3D eye position (between eyes)
    hR = cv2.Rodrigues(hr)[0]
    eye_center_3D = np.mean(face_model[[0, 1, 2, 3]], axis=0)  # Use actual face model eye points
    eye_center_camera = hR @ eye_center_3D.reshape(3, 1) + ht

    # 4. Define screen plane (adjust these values!)
    SCREEN = {
        "width_mm": 520,        # Physical screen width
        "height_mm": 320,       # Physical screen height
        "distance_mm": 600,     # Eye-to-screen distance
        "pixel_width": 1920,    # Screen resolution
        "pixel_height": 1080
    }

    # 5. Calculate gaze intersection
    t = (SCREEN["distance_mm"] - eye_center_camera[2][0]) / gaze_vector_camera[2][0]
    intersection = eye_center_camera + t * gaze_vector_camera

    # 2. Convert to screen coordinates (in millimeters)
    x_mm = intersection[0][0]
    y_mm = intersection[1][0]

    # 3. Convert mm to pixels (centered at screen center)
    x_pixel = int((x_mm / SCREEN["width_mm"]) * SCREEN["pixel_width"] + SCREEN["pixel_width"] / 2)
    y_pixel = int((y_mm / SCREEN["height_mm"]) * SCREEN["pixel_height"] + SCREEN["pixel_height"] / 2)

    # 4. Clamp to screen bounds
    x_pixel = np.clip(x_pixel, 0, SCREEN["pixel_width"] - 1)
    y_pixel = np.clip(y_pixel, 0, SCREEN["pixel_height"] - 1)
    on_screen = (0 <= x_pixel <= SCREEN["pixel_width"]) and (0 <= y_pixel <= SCREEN["pixel_height"])
    # --------------------------
    # DEBUG VISUALIZATION
    # --------------------------
    # Draw 3D gaze vector in original image
    # --------------------------
    # CORRECTED DEBUG VISUALIZATION
    # --------------------------
     # Create screen visualization image
    screen_img = np.full((SCREEN["pixel_height"], SCREEN["pixel_width"], 3), 255, dtype=np.uint8)  # White background

    # Draw screen border
    cv2.rectangle(screen_img, (0, 0), 
                (SCREEN["pixel_width"]-1, SCREEN["pixel_height"]-1),
                (0, 0, 0), 2)

    # Calculate dot position (clamp to screen edges if out of bounds)
    dot_x = int(np.clip(x_pixel, 0, SCREEN["pixel_width"]-1))
    dot_y = int(np.clip(y_pixel, 0, SCREEN["pixel_height"]-1))

    # Draw red dot with different intensity based on screen presence
   # Replace the screen visualization section with this code:

with mss.mss() as sct:
    # Get information for monitor 1 (primary monitor)
    monitor = sct.monitors[1]
    
    # Capture screen
    screenshot = sct.grab(monitor)
    
    # Convert to numpy array and BGR format
    screen_img = np.array(screenshot)
    screen_img = cv2.cvtColor(screen_img, cv2.COLOR_BGRA2BGR)
    
    # Resize to match our screen parameters (preserve aspect ratio)
    screen_img = cv2.resize(screen_img, (SCREEN["pixel_width"], SCREEN["pixel_height"]))
    
    # Create heatmap layer
    heatmap = np.zeros((screen_img.shape[0], screen_img.shape[1]), dtype=np.float32)
    
    # Create radial gradient for heatmap
    y, x = np.indices((screen_img.shape[0], screen_img.shape[1]))
    dx = x - dot_x
    dy = y - dot_y
    distance = np.sqrt(dx*dx + dy*dy)
    
    # Set heatmap intensity (inverse square law)
    intensity = 1.0 / (1.0 + distance/50)  # Adjust 50 for spread size
    heatmap = np.clip(intensity, 0, 1)
    
    # Apply Gaussian blur for smoothness
    heatmap = cv2.GaussianBlur(heatmap, (99, 99), 0)
    
    # Convert heatmap to color (Jet colormap)
    heatmap_color = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    
    # Blend heatmap with screenshot
    alpha = 0.5  # Adjust transparency (0-1)
    blended = cv2.addWeighted(screen_img, 1-alpha, heatmap_color, alpha, 0)
    
    # Add gaze point marker
    cv2.circle(blended, (dot_x, dot_y), 10, (0, 0, 255), -1)
    cv2.putText(blended, f"({dot_x}, {dot_y})", (dot_x+15, dot_y-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Save the result
    screen_output_path = 'example/output/screen_gaze_point.jpg'
    cv2.imwrite(screen_output_path, blended)

    # Add coordinate text
    cv2.putText(screen_img, f"({dot_x}, {dot_y})", (dot_x+15, dot_y-15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # Save screen visualization
    screen_output_path = 'example/output/screen_gaze_point.jpg'
    cv2.imwrite(screen_output_path, screen_img)
    print(f'Saved screen gaze visualization to: {screen_output_path}')

    # Project eye position to 2D (keep as NumPy array)
    eye_pos_2d, _ = cv2.projectPoints(eye_center_camera, 
                                    np.zeros(3), 
                                    np.zeros(3),
                                    camera_matrix, 
                                    camera_distortion)
    eye_pos_2d = eye_pos_2d[0][0].astype(int)  # Keep as NumPy array

    # Draw on original image
    draw_3d_gaze(image, eye_pos_2d, gaze_vector_camera)
    cv2.putText(image, f"Screen: {on_screen}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite('debug_visualization.jpg', image)

    print(f"Gaze screen coordinates: X={x_pixel:.1f}px, Y={y_pixel:.1f}px")
    print(f"Within screen bounds: {on_screen}")
    print(f"Gaze vector (camera space): {gaze_vector_camera.flatten()}")
    print(f"Eye position (camera space): {eye_center_camera.flatten()}")
    print(f"Intersection (camera space): {intersection.flatten()}")
    print(f"Raw x_pixel: {x_pixel}, Raw y_pixel: {y_pixel}")





with mss.mss() as sct:
    # Get information for monitor 1 (primary monitor)
    monitor = sct.monitors[1]
    
    # Capture screen
    screenshot = sct.grab(monitor)
    
    # Convert to numpy array and BGR format
    screen_img = np.array(screenshot)
    screen_img = cv2.cvtColor(screen_img, cv2.COLOR_BGRA2BGR)
    
    # Resize to match our screen parameters (preserve aspect ratio)
    screen_img = cv2.resize(screen_img, (SCREEN["pixel_width"], SCREEN["pixel_height"]))
    
    # Create heatmap layer
    heatmap = np.zeros((screen_img.shape[0], screen_img.shape[1]), dtype=np.float32)
    
    # Create radial gradient for heatmap
    y, x = np.indices((screen_img.shape[0], screen_img.shape[1]))
    dx = x - dot_x
    dy = y - dot_y
    distance = np.sqrt(dx*dx + dy*dy)
    
    # Set heatmap intensity (inverse square law)
    intensity = 1.0 / (1.0 + distance/50)  # Adjust 50 for spread size
    heatmap = np.clip(intensity, 0, 1)
    
    # Apply Gaussian blur for smoothness
    heatmap = cv2.GaussianBlur(heatmap, (99, 99), 0)
    
    # Convert heatmap to color (Jet colormap)
    heatmap_color = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    
    # Blend heatmap with screenshot
    alpha = 0.5  # Adjust transparency (0-1)
    blended = cv2.addWeighted(screen_img, 1-alpha, heatmap_color, alpha, 0)
    
    # Add gaze point marker
    cv2.circle(blended, (dot_x, dot_y), 3, (0, 0, 255), -1)
    cv2.putText(blended, f"({dot_x}, {dot_y})", (dot_x+15, dot_y-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Save the result
    screen_output_path = 'example/output/screen_gaze_point.jpg'
    cv2.imwrite(screen_output_path, blended)
