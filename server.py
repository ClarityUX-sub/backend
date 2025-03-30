import os
import subprocess
import cloudinary
import cloudinary.uploader
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cloudinary configuration
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

# Ensure absolute path for reliability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "example/output/screen_gaze_point.jpg")


@app.get("/analyze")
async def analyze():
    # Step 1: Generate the image by running d.py
    try:
        subprocess.run(["python", "d.py"], check=True)
    except subprocess.CalledProcessError as e:
        return JSONResponse(content={"error": f"Failed to run d.py: {str(e)}"}, status_code=500)

    # Step 2: Check if the image was successfully created
    if not os.path.exists(IMAGE_PATH):
        return JSONResponse(content={"error": "Image file not found"}, status_code=404)

    # Step 3: Upload image to Cloudinary
    try:
        upload_result = cloudinary.uploader.upload(IMAGE_PATH, public_id="screen_gaze_point")
        cloudinary_url = upload_result.get("secure_url")  # Get the secure URL

        if not cloudinary_url:
            return JSONResponse(content={"error": "Cloudinary upload failed, no URL returned"}, status_code=500)
        print("success") 
    except Exception as e:
        return JSONResponse(content={"error": f"Cloudinary upload failed: {str(e)}"}, status_code=500)

    # Step 4: Return the image URL to the frontend
    return JSONResponse(content={"cloudinary_url": cloudinary_url})
