
from datetime import datetime


def log(message: str):
    print('[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '] '+message)