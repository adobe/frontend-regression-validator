import json
from urllib.parse import urlparse
import re


class JavascriptLogPreprocessor(object):
    def __init__(self, log_file):
        self.logs = json.load(open(log_file, 'r'))
        self.js_messages = self._js_messages()

    def _js_messages(self):
        messages = []
        for log in self.logs:
            processed_message = re.sub(r'http:\S+', '', log['message'])
            messages.append(processed_message)
        return messages


class NetworkLogPreprocessor(object):
    def __init__(self, log_file):
        self.logs = json.load(open(log_file, 'r'))
        self.network_messages = self._network_messages()

    def _network_messages(self):
        messages = []
        for entry in self.logs['log']['entries']:
            request_text = entry['request']['url']
            request_path = urlparse(request_text).path
            response = entry['response']['status']
            message = request_path + str(response)

            messages.append(message)
        return messages
