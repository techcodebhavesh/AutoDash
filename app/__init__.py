from flask import Flask
from flask_cors import CORS
import os

# Absolute path to frontend build folder
frontend_build_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend', 'dist')

app = Flask(__name__, 
            static_folder=frontend_build_folder, 
            static_url_path='/')

CORS(app)

from app import views
