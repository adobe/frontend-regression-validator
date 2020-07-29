import unittest
import json
import time
from run import app
import uuid
import os

BASE_URL = 'http://127.0.0.1:5000/api'
TEST_BASELINE_URL = 'http://www.audi.ro'
TEST_UPDATED_URL = 'http://www.audi.ro'
TEST_CRAWL_DEPTH = '1'
TEST_MAX_URLS = '1'
TEST_PREFIX = 'test_prefix1'


def show_func_name(func):
    def echo_func(*func_args, **func_kwargs):
        print('')
        print('Running test {}'.format(func.__name__))
        return func(*func_args, **func_kwargs)

    return echo_func


class APITest(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    @show_func_name
    def test_post_new_job_inexistent_prefix(self):
        response = self.app.post(BASE_URL + '/verify',
                                 json={'baseline_url': TEST_BASELINE_URL, 'updated_url': TEST_UPDATED_URL,
                                       'max_depth': TEST_CRAWL_DEPTH,
                                       'max_urls': TEST_MAX_URLS, 'prefix': str(uuid.uuid4().hex)})
        data = json.loads(response.get_data())
        self.assertEqual(response.status_code, 200)
        self.assertIn("id", data)

    @show_func_name
    def test_post_new_job_existent_prefix(self):
        self.app.post(BASE_URL + '/verify',
                      json={'baseline_url': TEST_BASELINE_URL, 'updated_url': TEST_UPDATED_URL,
                            'max_depth': TEST_CRAWL_DEPTH,
                            'max_urls': TEST_MAX_URLS, 'prefix': TEST_PREFIX})
        response = self.app.post(BASE_URL + '/verify',
                                 json={'baseline_url': TEST_BASELINE_URL, 'updated_url': TEST_UPDATED_URL,
                                       'max_depth': TEST_CRAWL_DEPTH,
                                       'max_urls': TEST_MAX_URLS, 'prefix': TEST_PREFIX})
        self.assertEqual(response.status_code, 406)

    @show_func_name
    def test_post_incomplete_auth(self):
        response = self.app.post(BASE_URL + '/verify',
                                 json={'baseline_url': TEST_BASELINE_URL, 'updated_url': TEST_UPDATED_URL,
                                       'max_depth': TEST_CRAWL_DEPTH,
                                       'max_urls': TEST_MAX_URLS, 'prefix': str(uuid.uuid4().hex),
                                       'auth_baseline_password': 'a'})
        data = json.loads(response.get_data())
        self.assertEqual(response.status_code, 200)
        self.assertIn("id", data)

    @show_func_name
    def test_post_complete_auth(self):
        response = self.app.post(BASE_URL + '/verify',
                                 json={'baseline_url': TEST_BASELINE_URL, 'updated_url': TEST_UPDATED_URL,
                                       'max_depth': TEST_CRAWL_DEPTH,
                                       'max_urls': TEST_MAX_URLS, 'prefix': str(uuid.uuid4().hex),
                                       'auth_baseline_password': 'a',
                                       'auth_baseline_username': 'a',
                                       'auth_updated_password': 'b',
                                       'auth_updated_username': 'b'})
        data = json.loads(response.get_data())
        self.assertEqual(response.status_code, 200)
        self.assertIn("id", data)

    @show_func_name
    def test_get_all_jobs(self):
        self.app.post(BASE_URL + '/verify',
                      json={'baseline_url': TEST_BASELINE_URL, 'updated_url': TEST_UPDATED_URL,
                            'max_depth': TEST_CRAWL_DEPTH,
                            'max_urls': TEST_MAX_URLS, 'prefix': str(uuid.uuid4().hex)})
        response = self.app.get(BASE_URL + '/ids')
        data = json.loads(response.get_data())
        self.assertGreater(len(data), 0)

    @show_func_name
    def test_get_job(self):
        response = self.app.post(BASE_URL + '/verify',
                                 json={'baseline_url': TEST_BASELINE_URL, 'updated_url': TEST_UPDATED_URL,
                                       'max_depth': TEST_CRAWL_DEPTH,
                                       'max_urls': TEST_MAX_URLS, 'prefix': str(uuid.uuid4().hex)})
        data = json.loads(response.get_data())
        job_id = data['id']

        response = self.app.get(BASE_URL + '/verify?id={}'.format(job_id))
        data = json.loads(response.get_data())
        self.assertNotIn('Error', data)
        self.assertIn(data['Status'], ['Done', 'In progress', 'Failed'])

    @show_func_name
    def test_get_invalid_id(self):
        response = self.app.get(BASE_URL + '/verify?id=1234')
        self.assertEqual(response.status_code, 404)

    @show_func_name
    def test_get_result_valid_id(self):
        prefix = 'VALID_ID_TEST' + str(uuid.uuid4().hex)
        response = self.app.post(BASE_URL + '/verify',
                                 json={'baseline_url': TEST_BASELINE_URL, 'updated_url': TEST_UPDATED_URL,
                                       'max_depth': TEST_CRAWL_DEPTH,
                                       'max_urls': TEST_MAX_URLS, 'prefix': prefix})
        id = json.loads(response.get_data())['id']
        response = self.app.get(BASE_URL + '/verify?id={}'.format(id))
        status = json.loads(response.get_data())['Status']

        while status == 'In progress':
            response = self.app.get(BASE_URL + '/verify?id={}'.format(id))
            status = json.loads(response.get_data())['Status']
            time.sleep(1)
        self.assertEqual(status, 'Done')

        response = self.app.get(BASE_URL + '/result?id={}'.format(id))
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
