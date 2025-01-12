from src.lib.detection import *
from flask import Flask, render_template , Response
import shutil
import os
import sys

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  


@app.route('/login')
def login():
    return render_template('login.html')  


@app.route('/register')
def register():
    return render_template('register.html') 


@app.route('/Fotp')
def Fotp():
    return render_template('Fotp.html')  


@app.route('/projectinfo')
def projectinfo():
    return render_template('projectinfo.html')  


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Route to serve video stream
@app.route('/video_feed')
def video_feed():
    try:
        output_dir = os.path.join(os.getcwd(), 'output')
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
        upload_dir = os.path.join(os.getcwd(), 'upload')
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)

        # Change these paths ....
        return Response(detect_behaviour(
                video_file_path=r"C:\Users\arups\Downloads\videoplayback (1).mp4", 
                model_file_path_1=r"D:\Final-Year-Project\yolo11x-pose.pt",
                model_file_path_2=r"D:\Final-Year-Project\uit_model_final.pth"
            ),
                mimetype='multipart/x-mixed-replace; boundary=frame') 

    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

if __name__=="__main__":
    app.run(debug=True)
