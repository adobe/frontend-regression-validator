from flask import jsonify
from flask_restful import Resource
from endpoints.verify import id_to_urls, states
from utils.utils import get_time
from config.status_codes import STATUS_CODES


class IDList(Resource):
    def get(self):
        for id in states:
            status = states[id].poll()
            if status is None:
                id_to_urls[id]['status'] = 'In progress'
            elif status == 0:
                id_to_urls[id]['status'] = 'Done'
                id_to_urls[id]['stopped_at'] = get_time()
            else:
                id_to_urls[id]['status'] = STATUS_CODES.get(status, "Failed")
                id_to_urls[id]['stopped_at'] = get_time()
        return jsonify(id_to_urls)
