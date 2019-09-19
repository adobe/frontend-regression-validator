import json
import re


class JavascriptLogPreprocessor(object):
    def __init__(self, log_file):
        self.logs = json.load(open(log_file, 'r'))
        self.js_messages = self._js_messages()

    def _js_messages(self):
        messages = []
        for log in self.logs:
            processed_message = re.sub(r'http:\S+', '', log)
            messages.append(processed_message)
        return messages


class NetworkLogPreprocessor(object):
    def __init__(self, log_file):
        self.network_messages = json.load(open(log_file, 'r'))
