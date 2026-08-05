import ast
import re
import unittest
from pathlib import Path


def _admin_html() -> str:
    tree = ast.parse(Path(__file__).with_name('app.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'ADMIN_HTML'
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError('ADMIN_HTML not found')


def _app_source() -> str:
    return Path(__file__).with_name('app.py').read_text(encoding='utf-8')


ADMIN_HTML = _admin_html()
APP_SOURCE = _app_source()


class AdminPageTests(unittest.TestCase):
    def test_gallery_uses_in_page_tab_and_pagination(self):
        self.assertIn('data-tab="overview"', ADMIN_HTML)
        self.assertIn('data-tab="gallery"', ADMIN_HTML)
        self.assertIn('onclick="showGalleryTab()"', ADMIN_HTML)
        self.assertNotIn('onclick="loadMoreFiles()"', ADMIN_HTML)
        self.assertIn('id="gallery-page-size"', ADMIN_HTML)
        for size in (10, 20, 50, 100, 200):
            self.assertRegex(ADMIN_HTML, rf'<option value="{size}"')
        self.assertRegex(ADMIN_HTML, r'let galleryPageSize = 10;')
        self.assertIn('onclick="changeGalleryPage(-1)"', ADMIN_HTML)
        self.assertIn('onclick="changeGalleryPage(1)"', ADMIN_HTML)

    def test_gallery_requests_only_the_selected_page(self):
        self.assertIn('offset=${(galleryPage - 1) * galleryPageSize}', ADMIN_HTML)
        self.assertIn('limit=${galleryPageSize}', ADMIN_HTML)
        self.assertIn('galleryPage = 1;', ADMIN_HTML)
        self.assertNotIn('const fileLimit = 60;', ADMIN_HTML)

    def test_request_time_is_rendered_in_beijing_timezone(self):
        self.assertIn("timeZone: 'Asia/Shanghai'", ADMIN_HTML)
        self.assertIn('fmtBeijingTime(x.created_at)', ADMIN_HTML)
        self.assertNotIn("replace('+00:00','')", ADMIN_HTML)

    def test_gallery_tab_does_not_open_a_new_page(self):
        button = re.search(r'<button[^>]+onclick="showGalleryTab\(\)"[^>]*>', ADMIN_HTML)
        self.assertIsNotNone(button)
        self.assertNotIn('target=', button.group(0))
        self.assertNotIn('window.open', ADMIN_HTML)

    def test_requests_are_paginated_with_requested_page_sizes(self):
        self.assertIn('id="request-page-size"', ADMIN_HTML)
        for size in (10, 20, 50, 100, 200):
            self.assertRegex(ADMIN_HTML, rf'<option value="{size}"')
        self.assertIn('let requestPageSize = 10;', ADMIN_HTML)
        self.assertIn('offset=${(requestPage - 1) * requestPageSize}', ADMIN_HTML)
        self.assertIn('limit=${requestPageSize}', ADMIN_HTML)
        self.assertIn('onclick="changeRequestPage(-1)"', ADMIN_HTML)
        self.assertIn('onclick="changeRequestPage(1)"', ADMIN_HTML)
        self.assertNotIn('/api/admin/requests?limit=80', ADMIN_HTML)

    def test_complete_request_payload_is_persisted(self):
        self.assertIn('request_json TEXT', APP_SOURCE)
        self.assertIn('req.model_dump(mode="json")', APP_SOURCE)
        self.assertIn('"request_json": request_json', APP_SOURCE)
        self.assertIn('"request_json",', APP_SOURCE)

    def test_request_details_are_html_escaped(self):
        self.assertIn('function escapeHtml(value)', ADMIN_HTML)
        self.assertIn('escapeHtml(JSON.stringify(value, null, 2))', ADMIN_HTML)
        self.assertNotIn('<details class="request-details">', ADMIN_HTML)

    def test_task_detail_uses_an_in_page_tab_with_image_previews(self):
        self.assertIn('data-tab="task"', ADMIN_HTML)
        self.assertIn('id="task-tab"', ADMIN_HTML)
        self.assertIn('onclick="showTaskDetail(', ADMIN_HTML)
        self.assertIn("imageView('task-input-image'", ADMIN_HTML)
        self.assertIn("imageView('task-output-image'", ADMIN_HTML)
        self.assertIn('id="task-request-json"', ADMIN_HTML)
        self.assertIn('id="task-response-json"', ADMIN_HTML)
        self.assertNotIn('target="_blank"', ADMIN_HTML)

    def test_task_id_can_be_searched_and_copied(self):
        self.assertIn('id="task-search"', ADMIN_HTML)
        self.assertIn('function findTask()', ADMIN_HTML)
        self.assertIn('function copyTaskId()', ADMIN_HTML)
        self.assertIn('/api/admin/tasks/${encodeURIComponent(taskId)}', ADMIN_HTML)
        self.assertIn('x.task_id', ADMIN_HTML)

    def test_request_api_returns_page_metadata(self):
        self.assertIn('offset: int = Query(0, ge=0)', APP_SOURCE)
        self.assertIn('"total": total', APP_SOURCE)
        self.assertIn('"has_more": offset + len(rows) < total', APP_SOURCE)

    def test_task_tracking_schema_and_routes_exist(self):
        self.assertIn('task_id TEXT', APP_SOURCE)
        self.assertIn('response_json TEXT', APP_SOURCE)
        self.assertIn('@app.get("/api/tasks/{task_id}")', APP_SOURCE)
        self.assertIn('@app.get("/api/admin/tasks/{task_id}")', APP_SOURCE)
        self.assertIn('@app.exception_handler(RequestValidationError)', APP_SOURCE)


if __name__ == '__main__':
    unittest.main()
