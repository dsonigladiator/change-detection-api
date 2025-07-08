from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np
import cv2
import io
import base64

app = FastAPI()

# Helper: Convert image to base64
def image_to_base64(img_array):
    _, buffer = cv2.imencode('.png', img_array)
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64}"

# Main API Route
@app.post("/change-detection/")
async def change_detection(before: UploadFile = File(...), after: UploadFile = File(...)):
    # Load images
    before_img = Image.open(before.file).convert("RGB")
    after_img = Image.open(after.file).convert("RGB")

    before_np = np.array(before_img)
    after_np = np.array(after_img)

    # Resize after image to match before
    after_np = cv2.resize(after_np, (before_np.shape[1], before_np.shape[0]))

    # Difference Map (simple pixel difference)
    diff = cv2.absdiff(before_np, after_np)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    # Ensure normalized diff_gray is uint8
    norm_diff = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
    norm_diff = norm_diff.astype(np.uint8)

    diff_color = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)

    # Base64 images
    before_b64 = image_to_base64(before_np)
    after_b64 = image_to_base64(after_np)
    diff_b64 = image_to_base64(diff_color)

    # HTML slider for comparison
    html_content = f'''
    <html>
    <head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/noUiSlider/15.6.1/nouislider.min.css">
    </head>
    <body>
        <h2>Change Detection Slider</h2>
        <div id="container" style="position:relative;width:600px;height:400px;overflow:hidden;">
            <img id="before" src="{before_b64}" style="position:absolute;width:100%;height:auto;">
            <img id="after" src="{after_b64}" style="position:absolute;width:100%;height:auto;clip:rect(0px, 300px, 400px, 0px);">
        </div>
        <input id="slider" type="range" min="0" max="600" value="300" style="width:600px;">

        <script>
            const slider = document.getElementById('slider');
            const afterImg = document.getElementById('after');
            slider.addEventListener('input', () => {{
                let val = slider.value;
                afterImg.style.clip = `rect(0px, ${val}px, 400px, 0px)`;
            }});
        </script>

        <h3>Change Map:</h3>
        <img src="{diff_b64}" width="600">
    </body>
    </html>
    '''

    return HTMLResponse(content=html_content)
