from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2
import base64

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Change Detection API</title>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
            <style>
                body {
                    background: #f6f8fa;
                    font-family: 'Inter', Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                }
                .card {
                    background: #fff;
                    max-width: 520px;
                    margin: 64px auto 0 auto;
                    padding: 40px 36px 32px 36px;
                    box-shadow: 0 8px 32px rgba(30,34,90,0.10), 0 1.5px 5px rgba(0,0,0,0.04);
                    border-radius: 20px;
                    text-align: center;
                }
                h1 {
                    font-size: 2.1rem;
                    font-weight: 700;
                    margin: 0 0 12px 0;
                    letter-spacing: -1px;
                    color: #1a2533;
                }
                h2 {
                    margin: 32px 0 8px 0;
                    font-size: 1.2rem;
                    font-weight: 600;
                    color: #2554d6;
                }
                ul {
                    text-align: left;
                    margin: 12px 0 12px 20px;
                    padding: 0;
                    font-size: 1rem;
                    color: #283046;
                }
                li {
                    margin: 0 0 6px 0;
                }
                label {
                    font-weight: 500;
                    font-size: 1rem;
                    color: #222;
                }
                input[type="file"] {
                    margin: 8px 0 18px 0;
                }
                button {
                    font-family: inherit;
                    font-size: 1rem;
                    font-weight: 600;
                    background: #2554d6;
                    color: #fff;
                    border: none;
                    border-radius: 8px;
                    padding: 8px 22px;
                    margin-top: 6px;
                    margin-bottom: 8px;
                    cursor: pointer;
                    transition: background 0.18s;
                }
                button:hover {
                    background: #16337e;
                }
                pre {
                    background: #f1f3f6;
                    border-radius: 7px;
                    font-size: 0.98rem;
                    padding: 12px;
                    text-align: left;
                    overflow-x: auto;
                    color: #18223a;
                    margin-bottom: 10px;
                }
                .icon {
                    color: #2554d6;
                    margin-right: 8px;
                    font-size: 1.2em;
                    vertical-align: -2px;
                }
                .footer {
                    margin-top: 34px;
                    font-size: 0.98rem;
                    color: #98a4bc;
                    letter-spacing: 0.1em;
                }
                @media (max-width: 600px) {
                    .card { padding: 18px 7vw 24px 7vw; }
                    h1 { font-size: 1.32rem; }
                }
            </style>
        </head>
        <body>
            <div class="card">
                <span style="font-size:2.1rem;"><i class="fa-solid fa-layer-group icon"></i></span>
                <h1>Change Detection API</h1>
                <p style="font-size:1.06rem;color:#2d3753; margin: 12px 0 22px 0;">
                    Upload two images and visualize their detected changes, instantly.
                </p>
                <h2><i class="fa-solid fa-plug icon"></i>How to Use:</h2>
                <ul>
                    <li>
                        <b>POST</b> to <code>/change-detection/</code> with form-data containing:<br>
                        <span style="margin-left:1.1em;"><b>before</b>: the first image file</span><br>
                        <span style="margin-left:1.1em;"><b>after</b>: the second image file</span>
                    </li>
                </ul>
                <h2><i class="fa-solid fa-terminal icon"></i>Sample <code>curl</code> command:</h2>
                <pre>
curl -X POST {your_url}/change-detection/ \\
  -F "before=@before.jpg" -F "after=@after.jpg"
                </pre>
                <h2><i class="fa-solid fa-upload icon"></i>Sample Upload Form:</h2>
                <form action="/change-detection/" method="post" enctype="multipart/form-data" target="_blank">
                  <label>Before image:<br><input type="file" name="before" required /></label><br>
                  <label>After image:<br><input type="file" name="after" required /></label><br>
                  <button type="submit"><i class="fa-solid fa-bolt"></i> Detect Change</button>
                </form>
                <div class="footer">
                    Built with <span style="color:#ea4687;font-size:1.15em;">&#10084;&#65039;</span> using FastAPI
                </div>
            </div>
        </body>
    </html>
    """


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
        <title>Change Detection Result</title>
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
