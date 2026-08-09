import importlib.util
import base64
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from contextlib import closing
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

    def test_cached_request_remains_visible_when_filtered_by_parent_task(self):
        self.module._RESPONSE_CACHE.clear()
        processed = {
            'image_bytes': b'cached-image',
            'face_count': 1,
            'elapsed_ms': 4.0,
            'faces': [],
        }
        image_url = 'https://example.com/cache-parent.jpg'
        with patch.object(self.module, '_download', return_value=b'input-image'), \
                patch.object(self.module, 'process_image', return_value=processed):
            first = self.client.post(
                '/api/face_blur',
                json={'image_url': image_url, 'mode': 'gaussian', 'parent_task_id': 'batch-original'},
            )
        self.assertEqual(first.status_code, 200)
        output_file = first.json()['output_file']

        cached = self.client.post(
            '/api/face_blur',
            json={'image_url': image_url, 'mode': 'gaussian', 'parent_task_id': 'batch-cached'},
        )
        self.assertEqual(cached.status_code, 200)
        self.assertEqual(cached.json()['output_file'], output_file)
        cached_task_id = cached.json()['task_id']

        files = self.client.get(
            '/api/admin/files?offset=0&limit=10&parent_task_id=batch-cached',
            headers={'X-Admin-Token': 'test-admin'},
        )
        self.assertEqual(files.status_code, 200)
        self.assertEqual(files.json()['total'], 1)
        self.assertEqual(files.json()['items'][0]['name'], output_file)
        self.assertEqual(files.json()['items'][0]['task_id'], cached_task_id)

    def test_global_blur_defaults_apply_only_when_request_omits_them(self):
        processed = {
            'image_bytes': b'global-settings-image',
            'face_count': 1,
            'elapsed_ms': 4.0,
            'faces': [],
        }
        settings = {
            'score_threshold': 0.61,
            'expand_ratio': 0.40,
            'min_face_skip': 75,
            'dot_radius': 4,
            'face_grid_step': 16,
            'grid_n': 7,
        }
        saved = self.client.post(
            '/api/admin/settings',
            headers={'X-Admin-Token': 'test-admin'},
            json=settings,
        )
        self.assertEqual(saved.status_code, 200)

        with patch.object(self.module, '_download', return_value=b'input-image'), \
                patch.object(self.module, 'process_image', return_value=processed) as process_mock:
            defaulted = self.client.post(
                '/api/face_blur',
                json={'image_url': 'https://example.com/global-default.jpg', 'mode': 'landmark_whole_face'},
            )
            explicit = self.client.post(
                '/api/face_blur',
                json={
                    'image_url': 'https://example.com/explicit-default.jpg',
                    'mode': 'landmark_whole_face',
                    'score_threshold': 0.45,
                    'min_face_skip': 20,
                },
            )
        self.assertEqual(defaulted.status_code, 200)
        self.assertEqual(explicit.status_code, 200)
        first_kwargs = process_mock.call_args_list[0].kwargs
        second_kwargs = process_mock.call_args_list[1].kwargs
        self.assertEqual(first_kwargs['score_threshold'], 0.61)
        self.assertEqual(first_kwargs['min_face_skip'], 75)
        self.assertEqual(first_kwargs['face_grid_step'], 16)
        self.assertEqual(first_kwargs['grid_n'], 7)
        self.assertFalse(first_kwargs['adaptive'])
        self.assertEqual(second_kwargs['score_threshold'], 0.45)
        self.assertEqual(second_kwargs['min_face_skip'], 20)

        with patch.object(self.module, '_download', return_value=b'input-image'), \
                patch.object(self.module, 'process_image', return_value=processed) as landmark_mock:
            response = self.client.post(
                '/api/face_blur',
                json={'image_url': 'https://example.com/landmark-spacing.jpg', 'mode': 'landmark'},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(landmark_mock.call_args.kwargs['spacing'], 16)

        loaded = self.client.get(
            '/api/admin/settings', headers={'X-Admin-Token': 'test-admin'}
        )
        self.assertEqual(loaded.json()['settings']['grid_n'], 7)
        self.client.post(
            '/api/admin/settings',
            headers={'X-Admin-Token': 'test-admin'},
            json={
                'score_threshold': 0.52,
                'expand_ratio': 0.30,
                'min_face_skip': 50,
                'dot_radius': 3,
                'face_grid_step': 14,
                'grid_n': 5,
            },
        )

    def test_multi_mode_profiles_reach_blur_core(self):
        processed = {'image_bytes': b'profile-image', 'face_count': 1, 'elapsed_ms': 1.0, 'faces': []}
        profiles = [{'name': 'medium', 'min_width': 100, 'max_width': 199,
                     'modes': ['gaussian', 'landmark_whole_face'],
                     'face_grid_step': 11, 'dot_radius': 2, 'grid_n': 4}]
        with patch.object(self.module, '_download', return_value=b'input-image'), \
                patch.object(self.module, 'process_image', return_value=processed) as process_mock:
            response = self.client.post('/api/face_blur', json={
                'image_url': 'https://example.com/profile.jpg',
                'mode': 'landmark_whole_face',
                'modes': ['gaussian', 'landmark_whole_face'],
                'face_profiles': profiles,
            })
        self.assertEqual(response.status_code, 200)
        kwargs = process_mock.call_args.kwargs
        self.assertEqual(kwargs['modes'], ['gaussian', 'landmark_whole_face'])
        self.assertEqual(kwargs['face_profiles'], profiles)

    def test_clear_cache_removes_memory_cache(self):
        self.module._cache_set('test-key', {'ok': True})
        old_epoch = self.module._get_setting('cache_epoch', '0')
        response = self.client.post(
            '/api/admin/clear-cache', headers={'X-Admin-Token': 'test-admin'}
        )
        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(response.json()['cleared_l1'], 1)
        self.assertIsNone(self.module._cache_get('test-key'))
        self.assertNotEqual(self.module._get_setting('cache_epoch', '0'), old_epoch)

    def test_clear_cache_by_parent_only_removes_matching_l1_and_l2_entries(self):
        parent_file = 'parent-cache-output.jpg'
        other_file = 'other-cache-output.jpg'
        (self.module.STATIC_DIR / parent_file).write_bytes(b'parent')
        (self.module.STATIC_DIR / other_file).write_bytes(b'other')
        self.module._insert_request({
            'task_id': 'parent-cache-task', 'status': 'ok', 'mode': 'gaussian',
            'output_file': parent_file, 'output_url': '/static/' + parent_file,
            'parent_task_id': 'parent-to-clear',
        })
        self.module._insert_request({
            'task_id': 'other-cache-task', 'status': 'ok', 'mode': 'gaussian',
            'output_file': other_file, 'output_url': '/static/' + other_file,
            'parent_task_id': 'parent-to-keep',
        })
        self.module._cache_set('parent-l1', {'output_file': parent_file, 'output_url': '/static/' + parent_file})
        self.module._cache_set('other-l1', {'output_file': other_file, 'output_url': '/static/' + other_file})
        self.module._db_cache_set('parent-l2', 'https://example.com/parent', '/static/' + parent_file, parent_file, 'gaussian')
        self.module._db_cache_set('other-l2', 'https://example.com/other', '/static/' + other_file, other_file, 'gaussian')

        response = self.client.post(
            '/api/admin/clear-cache?parent_task_id=parent-to-clear',
            headers={'X-Admin-Token': 'test-admin'},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['cleared_l1'], 1)
        self.assertEqual(response.json()['cleared_l2'], 1)
        self.assertIsNone(self.module._cache_get('parent-l1'))
        self.assertIsNotNone(self.module._cache_get('other-l1'))
        self.assertIsNone(self.module._db_cache_get('parent-l2'))
        self.assertIsNotNone(self.module._db_cache_get('other-l2'))

    def test_default_download_limit_accepts_requested_large_image(self):
        self.assertGreaterEqual(self.module.MAX_IMAGE_BYTES, 21271569)

    def test_lab_returns_before_and_after_confidence_summary(self):
        before = {
            'image_bytes': b'blurred-image',
            'face_count': 2,
            'elapsed_ms': 4.0,
            'faces': [
                {'x': 180, 'y': 30, 'w': 40, 'h': 50, 'score': 0.91, 'landmarks': {}},
                {'x': 20, 'y': 10, 'w': 30, 'h': 35, 'score': 0.73, 'landmarks': {'nose': (30, 25)}},
            ],
        }
        after = {
            'image_bytes': b'probe-image',
            'face_count': 1,
            'elapsed_ms': 2.0,
            'faces': [{'score': 0.55}],
        }
        image_b64 = base64.b64encode(b'input-image').decode('ascii')
        with patch.object(sys.modules['face_blur'], 'process_image', side_effect=[before, after]) as process_mock:
            response = self.client.post(
                '/api/lab/test',
                headers={'X-Admin-Token': 'test-admin'},
                json={'image_base64': image_b64, 'mode': 'landmark_whole_face', 'score_threshold': 0.52},
            )
        self.assertEqual(response.status_code, 200)
        output_url = response.json()['output_url']
        self.assertRegex(output_url, r'/i/[0-9a-f]{12}$')
        short_image = self.client.get(output_url)
        self.assertEqual(short_image.status_code, 200)
        self.assertEqual(short_image.content, b'blurred-image')
        confidence = response.json()['confidence']
        self.assertEqual(confidence['threshold'], 0.52)
        self.assertEqual(confidence['before']['face_count'], 2)
        self.assertEqual(confidence['before']['max_score'], 0.91)
        self.assertEqual(confidence['before']['avg_score'], 0.82)
        self.assertEqual(confidence['before']['faces'][0]['x'], 20)
        self.assertEqual(confidence['before']['faces'][0]['width'], 30)
        self.assertEqual(confidence['before']['faces'][0]['landmark_count'], 1)
        self.assertEqual(confidence['after']['face_count'], 1)
        self.assertEqual(confidence['after']['max_score'], 0.55)
        self.assertEqual(process_mock.call_count, 2)
        self.assertFalse(process_mock.call_args_list[0].kwargs.get('adaptive', True))

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
        with closing(sqlite3.connect(self.module.DB_PATH)) as conn:
            columns = {row[1] for row in conn.execute('PRAGMA table_info(requests)')}
        self.assertIn('task_id', columns)
        self.assertIn('response_json', columns)
        with closing(sqlite3.connect(self.module.DB_PATH)) as conn:
            indexes = {row[1] for row in conn.execute("PRAGMA index_list('requests')")}
        self.assertIn('idx_requests_parent_output', indexes)
        self.assertIn('idx_requests_parent_id', indexes)


if __name__ == '__main__':
    unittest.main()
