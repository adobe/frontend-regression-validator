from flask import Flask


class MyServer(Flask):

    def __init__(self, *args, **kwargs):
        super(MyServer, self).__init__(*args, **kwargs)

        # instanciate your variables here
        self.proxy = None
        self.server = None


app = MyServer(__name__, static_url_path='')
# app = Flask(__name__, static_url_path='')
