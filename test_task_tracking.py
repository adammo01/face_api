import importlib.util
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


class TaskTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        root = Path(cls.temp_dir.name)
        os.environ['FACE_BLUR_DB_PATH'] = str(root / 'faceblur.sqlite3')
        os.environ['FACE_BLUR_STATIC_DIR'] = str(root / 'static')
        os.environ['FACE_BLUR_ADMIN_TOKEN'] = 'test-admin'
        os.environ['FACE_BLUR_MAX_RETRIES'] = '0'

        fake_cv2 = types.ModuleType('cv2')
        fake_face_blur = types.ModuleType('face_blur')
        fake_face_blur.process_image = lambda *args, **kwargs: None
        sys.modules['cv2'] = fake_cv2
        sys.modules['face_blur'] = fake_face_blur

        app_path = Path(__file__).with_name('app.py')
        spec = importlib.util.spec_from_file_location('faceblur_test_app', app_path)
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        spec.loader.exec_module(cls.module)
        cls.client_context = TestClient(cls.module.app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)
        cls.temp_dir.cleanup()

    def test_success_response_has_queryable_task_id_and_full_audit_data(self):
        processed = {
            'image_bytes': b'finished-image',
            'face_count': 1,
            'elapsed_ms': 12.5,
            'faces': [],
        }
        headers = {
            'Authorization': 'Bearer secret-value',
            'X-Trace-Id': 'trace-123',
            'X-Custom-Token': 'custom-secret',
            'Cf-Access-Jwt-Assertion': 'access-secret',
        }
        with patch.object(self.module, '_download', return_value=b'input-image'), \
                patch.object(self.module, 'process_image', return_value=processed):
            response = self.client.post(
                '/api/face_blur?source=test-suite',
                headers=headers,
                json={'image_url': 'https://example.com/input.jpg', 'mode': 'gaussian'},
            )

        self.assertEqual(response.status_code, 200)
        task_id = response.json()['task_id']
        self.assertRegex(task_id, r'^[0-9a-f]{32}$')

        public_status = self.client.get(f'/api/tasks/{task_id}')
        self.assertEqual(public_status.status_code, 200)
        self.assertEqual(public_status.json()['status'], 'ok')
        self.assertEqual(public_status.json()['task_id'], task_id)
        self.assertIn('output_url', public_status.json())

        detail = self.client.get(
            f'/api/admin/tasks/{task_id}', headers={'X-Admin-Token': 'test-admin'}
        )
        self.assertEqual(detail.status_code, 200)
        record = detail.json()['task']
        self.assertEqual(record['request']['method'], 'POST')
        self.assertEqual(record['request']['path'], '/api/face_blur')
        self.assertEqual(record['request']['query']['source'], 'test-suite')
        self.assertEqual(record['request']['headers']['authorization'], '[REDACTED]')
        self.assertEqual(record['request']['headers']['x-custom-token'], '[REDACTED]')
        self.assertEqual(record['request']['headers']['cf-access-jwt-assertion'], '[REDACTED]')
        self.assertEqual(record['request']['headers']['x-trace-id'], 'trace-123')
        self.assertEqual(record['request']['body']['mode'], 'gaussian')
        self.assertEqual(record['response']['task_id'], task_id)

    def test_failed_request_keeps_task_id_and_can_be_located(self):
        with patch.object(self.module, '_download', side_effect=OSError('upstream unavailable')):
            response = self.client.post(
                '/api/face_blur',
                json={'image_url': 'https://example.com/missing.jpg', 'mode': 'solid'},
            )

        self.assertEqual(response.status_code, 400)
        task_id = response.json()['task_id']
        status = self.client.get(f'/api/tasks/{task_id}')
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()['status'], 'download_error')
        self.assertIn('upstream unavailable', status.json()['error'])

    def test_validation_failure_is_recorded_with_task_id(self):
        response = self.client.post(
            '/api/face_blur?source=invalid-test',
            headers={'X-Trace-Id': 'invalid-trace'},
            json={'image_url': 'https://example.com/input.jpg', 'mode': 'unknown'},
        )

        self.assertEqual(response.status_code, 422)
        task_id = response.json()['task_id']
        status = self.client.get(f'/api/tasks/{task_id}').json()
        self.assertEqual(status['status'], 'validation_error')
        detail = self.client.get(
            f'/api/admin/tasks/{task_id}', headers={'X-Admin-Token': 'test-admin'}
        ).json()['task']
        self.assertEqual(detail['request']['body']['mode'], 'unknown')
        self.assertEqual(detail['request']['query']['source'], 'invalid-test')

    def test_concurrency_rejection_is_recorded_with_task_id(self):
        with patch.object(self.module, '_get_int_setting', return_value=1), \
                patch.object(self.module, '_inflight_try_acquire', return_value=False):
            response = self.client.post(
                '/api/face_blur',
                json={'image_url': 'https://example.com/input.jpg', 'mode': 'solid'},
            )

        self.assertEqual(response.status_code, 429)
        task_id = response.json()['task_id']
        status = self.client.get(f'/api/tasks/{task_id}').json()
        self.assertEqual(status['status'], 'rejected')

    def test_database_migration_adds_task_columns(self):
        with sqlite3.connect(self.module.DB_PATH) as conn:
            columns = {row[1] for row in conn.execute('PRAGMA table_info(requests)')}
        self.assertIn('task_id', columns)
        self.assertIn('response_json', columns)


if __name__ == '__main__':
    unittest.main()
