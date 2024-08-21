import uuid
import os

NGINX_FOLDER = os.environ['NGINX_FOLDER']
NGINX_URL = "http://localhost/content/"


def getNginxPath(file):
    return NGINX_FOLDER+file


def getNginxUrl(file):
    return NGINX_URL+file


def getFile():
    return str(uuid.uuid4())+".png"
