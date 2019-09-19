from flask import request
from flask_restful import Resource
from subprocess import Popen
import uuid
from utils.utils import get_time, check_unique_prefix
from config.status_codes import STATUS_CODES
from config.proxy import get_proxy
import socket
from urllib.parse import urlparse

states = {}
id_to_urls = {}


class Verify(Resource):
    def __init__(self, app):
        self.app = app

    def post(self):
        baseline_url = request.json['baseline_url']
        updated_url = request.json['updated_url']
        proxy = get_proxy(self.app)
        baseline_url_host = urlparse(baseline_url)
        updated_url_host = urlparse(updated_url)
        print(baseline_url_host.netloc)
        print(socket.gethostbyname(baseline_url))
        # proxy.remap_hosts(baseline_url_host.netloc, socket.gethostbyname(baseline_url))
        # proxy.remap_hosts(updated_url_host.netloc, socket.gethostbyname(updated_url))
        proxy.remap_hosts(baseline_url_host.netloc, 'www.google.com')
        proxy.remap_hosts(updated_url_host.netloc, 'www.google.com')
        max_depth = request.json['max_depth']
        max_urls = request.json['max_urls']
        prefix = request.json['prefix']
        auth_baseline_username = request.json.get('auth_baseline_username', '')
        auth_baseline_password = request.json.get('auth_baseline_password', '')
        auth_updated_username = request.json.get('auth_updated_username', '')
        auth_updated_password = request.json.get('auth_updated_password', '')

        if not check_unique_prefix(prefix, id_to_urls):
            return {'Error': 'Please choose a different prefix'}, 406

        p = Popen(
            ['python3', 'worker_crawl.py', '--baseline-url', baseline_url, '--updated-url', updated_url,
             '--max-depth',
             max_depth, '--max-urls', max_urls, '--prefix', prefix, '--auth-baseline-username', auth_baseline_username,
             '--auth-baseline-password', auth_baseline_password, '--auth-updated-username', auth_updated_username,
             '--auth-updated-password', auth_updated_password, '--proxy-host', proxy.host, '--proxy-port',
             str(proxy.port)])
        if p.poll() is not None and p.poll() > 0:
            return {'Error': 'Failed to launch crawler'}, 406
        id = str(uuid.uuid4().hex)
        states[id] = p

        id_to_urls[id] = {'baseline_url': baseline_url, 'updated_url': updated_url, 'status': 'Starting',
                          'started_at': get_time(), 'stopped_at': 'None', 'prefix': prefix, 'max_depth': max_depth,
                          'max_urls': max_urls, 'auth_baseline_username': auth_baseline_username,
                          'auth_baseline_password': auth_baseline_password,
                          'auth_updated_username': auth_updated_username,
                          'auth_updated_password': auth_updated_password}
        return {'id': id}, 200

    def get(self):
        id = request.args.get('id')
        if id not in states:
            return {'Error': 'Invalid ID'}, 404

        status = states[id].poll()

        if status is None:
            id_to_urls[id]['status'] = 'In progress'
            return {'Status': id_to_urls[id]['status']}, 200
        elif status == 0:
            id_to_urls[id]['status'] = 'Done'
            id_to_urls[id]['stopped_at'] = get_time()
            return {'Status': id_to_urls[id]['status']}, 200
        else:
            id_to_urls[id]['status'] = STATUS_CODES.get(status, "Failed")
            id_to_urls[id]['stopped_at'] = get_time()
            return {'Status': id_to_urls[id]['status']}, 200
