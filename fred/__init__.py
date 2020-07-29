from flask import Flask
import logging

logging.basicConfig(level=logging.DEBUG, format="[%(levelname)8s | %(asctime)s | %(filename)-20s:%(lineno)3s | %(funcName)-26s] %(message)s")#"%(asctime)s:%(levelname)s:%(message)s")

app = Flask(__name__, static_url_path='')
