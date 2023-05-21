from flask import Flask, render_template, request, current_app
from werkzeug.utils import secure_filename
from utils import *


app = Flask(__name__)

@app.route('/')
def uploader_file():
    return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        save_path = 'static/' + secure_filename(f.filename)
        current_app.current_image = save_path
        f.save(save_path)
        return render_template('intro.html', input_image = save_path)

@app.route('/detect', methods = ['GET', 'POST'])
def detectar():
    if request.method == 'POST':
        image = current_app.current_image

        detected = detect_and_save(model, image, category_index)

        return render_template('detection.html', input_image=detected)

if __name__=='__main__':
    app.run(debug=True)