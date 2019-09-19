from config.proxy import get_server, init

from flask import Flask
from flask_restful import Api
from flask_cors import CORS
from endpoints.verify import Verify
from endpoints.get_ids import IDList
from endpoints.shutdown import Shutdown
from endpoints.verify import states, id_to_urls
from endpoints.get_result import Result
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, request, send_from_directory
import sys

sys.path.append('../')
from fred import app


def clear_ended():
    for k in states.copy():
        if states[k].poll() is not None:
            del states[k]
            del id_to_urls[k]


@app.route('/static/<path:path>')
def send_js(path):
    return send_from_directory('frontend', path)


if __name__ == '__main__':
    if app.config["DEBUG"]:
        @app.after_request
        def after_request(response):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
            response.headers["Expires"] = 0
            response.headers["Pragma"] = "no-cache"
            return response
    cors = CORS(app, resources={r"*": {"origins": "*"}})
    # print(hex(id(app)))
    api = Api(app)
    api.add_resource(Verify, "/api/verify", resource_class_kwargs={'app': app})
    api.add_resource(IDList, "/api/ids")
    api.add_resource(Shutdown, "/api/shutdown")
    api.add_resource(Result, "/api/result")
    init(app)

    app.run(host='0.0.0.0', debug=True)
    server = get_server(app)
    server.stop()
