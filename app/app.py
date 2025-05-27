from flask import Flask, render_template, request, redirect, url_for, flash
import os
import sys
from werkzeug.utils import secure_filename

# Add parent directory to sys.path to import compiler module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from compiler import optimize_code

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flashing messages
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1MB max file size
app.config['UPLOAD_EXTENSIONS'] = ['.py']

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in app.config['UPLOAD_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        code = request.form.get('code')
        file = request.files.get('file')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            if not allowed_file(filename):
                flash('Invalid file type. Only .py files are allowed.')
                return render_template('index.html')
            try:
                code = file.read().decode('utf-8')
            except Exception:
                flash('Failed to read uploaded file. Please ensure it is a valid UTF-8 encoded Python file.')
                return render_template('index.html')
        if not code:
            flash("Please enter some Python code or upload a .py file to optimize.")
            return render_template('index.html')
        try:
            result = optimize_code(code)
            return render_template('result.html', result=result)
        except Exception as e:
            flash(f"Error during optimization: {e}")
            return render_template('index.html')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
