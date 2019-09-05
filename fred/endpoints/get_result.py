from flask import request
from flask_restful import Resource
import os
import json
from endpoints.verify import id_to_urls


class Result(Resource):
    def get(self):
        id = request.args.get('id')
        id_dict = id_to_urls.get(id, "none")

        if id_dict == "none":
            return {'Error': 'Report does not exist'}, 404

        prefix = id_dict['prefix']
        report_path = os.path.join('./tmp', prefix + '_report.json')

        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)

                return report, 200
        return {'Error': 'Report does not exist'}, 404
