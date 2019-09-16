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


def clear_ended():
    for k in states.copy():
        if states[k].poll() is not None:
            del states[k]
            del id_to_urls[k]


app = Flask(__name__, static_url_path='')


@app.route('/static/<path:path>')
def send_js(path):
    return send_from_directory('frontend', path)


if app.config["DEBUG"]:
    @app.after_request
    def after_request(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
        response.headers["Expires"] = 0
        response.headers["Pragma"] = "no-cache"
        return response

cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)
api.add_resource(Verify, "/api/verify")
api.add_resource(IDList, "/api/ids")
api.add_resource(Shutdown, "/api/shutdown")
api.add_resource(Result, "/api/result")
# scheduler = BackgroundScheduler()
# scheduler.add_job(func=clear_ended, trigger='interval', seconds=300)
# scheduler.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
