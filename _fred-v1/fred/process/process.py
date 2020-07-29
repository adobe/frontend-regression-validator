from data.log_preprocess import JavascriptLogPreprocessor, NetworkLogPreprocessor
from utils.utils import intersection
from scores.scores import Scores


class LogProcessor(object):
    def __init__(self, baseline_js_log_file, updated_js_log_file, baseline_network_log_file, updated_network_log_file):
        self.baseline_js_logs = JavascriptLogPreprocessor(baseline_js_log_file)
        self.updated_js_logs = JavascriptLogPreprocessor(updated_js_log_file)
        self.baseline_network_logs = NetworkLogPreprocessor(baseline_network_log_file)
        self.updated_network_logs = NetworkLogPreprocessor(updated_network_log_file)
        self.result = {}

    def run(self):
        js_logs = {'in_baseline': len(self.baseline_js_logs.js_messages),
                   'in_upgraded': len(self.updated_js_logs.js_messages),
                   'in_both': len(
                       intersection(self.baseline_js_logs.js_messages, self.updated_js_logs.js_messages))}

        network_logs = {'in_baseline': len(self.baseline_network_logs.network_messages),
                        'in_upgraded': len(self.updated_network_logs.network_messages),
                        'in_both': len(
                            intersection(self.baseline_network_logs.network_messages,
                                         self.updated_network_logs.network_messages))}

        risk_score_js = Scores.logs_divergence(self.baseline_js_logs.js_messages, self.updated_js_logs.js_messages)

        risk_score_network = Scores.logs_divergence(self.baseline_network_logs.network_messages,
                                                    self.updated_network_logs.network_messages)
        risk_score = max(risk_score_js, risk_score_network)
        self.result = {'javascript': js_logs, 'network': network_logs, 'risk_score': risk_score}
        return self.result
