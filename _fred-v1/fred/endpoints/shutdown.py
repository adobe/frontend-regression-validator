from flask_restful import Resource
from flask import request


class Shutdown(Resource):
    def get(self):
        shutdown = request.environ.get('werkzeug.server.shutdown')
        if shutdown is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        shutdown()
        return 'Server shutting down'
