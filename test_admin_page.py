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
        self.assertIn("if(activeParentSearch) url += '&parent_task_id=' + encodeURIComponent(activeParentSearch);", ADMIN_HTML)
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
        self.assertIn('let requestPageSize = 20;', ADMIN_HTML)
        self.assertIn("'&limit=' + requestPageSize", ADMIN_HTML)
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

    def test_admin_token_is_forwarded_to_lab_and_lab_initializes_after_dom(self):
        self.assertIn('function showLabTab()', ADMIN_HTML)
        self.assertIn('lab-frame', ADMIN_HTML)
        self.assertIn('window.addEventListener("DOMContentLoaded"', APP_SOURCE)

    def test_lab_images_open_in_an_enlarged_preview(self):
        self.assertIn('onclick="labOpenPreview(this)"', APP_SOURCE)
        self.assertIn('id="lab-image-modal"', APP_SOURCE)
        self.assertIn('function labOpenPreview(image)', APP_SOURCE)
        self.assertIn('if(event.key === "Escape") labClosePreview();', APP_SOURCE)

    def test_lab_supports_saved_parameter_presets_and_defaults(self):
        self.assertIn('id="lab-preset"', APP_SOURCE)
        self.assertIn('onclick="labSavePreset()"', APP_SOURCE)
        self.assertIn('onclick="labDeletePreset()"', APP_SOURCE)
        self.assertIn('onclick="labResetDefaults()"', APP_SOURCE)
        self.assertIn('const LAB_PRESETS_KEY = "faceblur.lab.presets.v1";', APP_SOURCE)
        self.assertIn('localStorage.setItem(LAB_PRESETS_KEY', APP_SOURCE)
        self.assertIn('labSetParams(LAB_DEFAULTS);', APP_SOURCE)

    def test_lab_supports_multi_mode_distance_profiles(self):
        self.assertIn('id="lab-mode" class="lab-mode-options"', APP_SOURCE)
        self.assertIn('type="checkbox" value="gaussian"', APP_SOURCE)
        self.assertIn('function labSelectedModes()', APP_SOURCE)
        self.assertIn('face_profiles', APP_SOURCE)
        self.assertIn('modes: params.modes', APP_SOURCE)
        self.assertIn('function labFillProfileExample()', APP_SOURCE)
        self.assertIn('查看填写示例', APP_SOURCE)

    def test_lab_shows_before_and_after_face_confidence(self):
        self.assertIn('id="lab-confidence-before"', APP_SOURCE)
        self.assertIn('id="lab-confidence-after"', APP_SOURCE)
        self.assertIn('d.confidence.before', APP_SOURCE)
        self.assertIn('d.confidence.after', APP_SOURCE)
        self.assertIn('"confidence": {"threshold": req.score_threshold', APP_SOURCE)

    def test_lab_shows_per_face_detection_details(self):
        self.assertIn('id="lab-detection-details"', APP_SOURCE)
        self.assertIn('function labRenderDetectionDetails(', APP_SOURCE)
        self.assertIn('从左到右', APP_SOURCE)
        self.assertIn('"landmark_count"', APP_SOURCE)

    def test_lab_runs_are_explicitly_unique_and_uncached(self):
        self.assertIn('opts.cache = "no-store"', APP_SOURCE)
        self.assertIn('X-Lab-Run-Id', APP_SOURCE)
        self.assertIn('"task_id": task_id', APP_SOURCE)
        self.assertIn('已重新处理 · 任务 ID:', APP_SOURCE)

    def test_lab_returns_a_public_short_link_for_output(self):
        self.assertIn('@app.get("/i/{image_id}")', APP_SOURCE)
        self.assertIn('def _short_url_for(', APP_SOURCE)
        self.assertIn('id="lab-output-link"', APP_SOURCE)
        self.assertIn('function labCopyOutputLink()', APP_SOURCE)

    def test_lab_defaults_do_not_override_selected_global_mode(self):
        self.assertIn('const LAB_DEFAULTS = {mode:"landmark_whole_face", modes:["landmark_whole_face"], face_profiles:[]', APP_SOURCE)
        self.assertIn('function labModeChanged(input)', APP_SOURCE)
        self.assertIn('id="lab-profiles" rows="8"', APP_SOURCE)
        self.assertIn('>[]</textarea>', APP_SOURCE)

    def test_lab_profile_example_uses_three_non_overlapping_ranges(self):
        self.assertIn('min_width:40,max_width:59', APP_SOURCE)
        self.assertIn('min_width:60,max_width:79', APP_SOURCE)
        self.assertIn('min_width:80,max_width:110', APP_SOURCE)
        self.assertIn('name:"large"', APP_SOURCE)

    def test_distance_profile_docs_page_is_linked_and_public(self):
        self.assertIn('@app.get("/docs/face-profiles"', APP_SOURCE)
        self.assertIn('href="/docs/face-profiles"', APP_SOURCE)
        self.assertIn('返回打码实验室', APP_SOURCE)

    def test_global_settings_has_one_clear_cache_button(self):
        self.assertEqual(ADMIN_HTML.count('onclick="clearCache()"'), 1)

    def test_parent_cache_clear_control_is_available(self):
        self.assertIn('function clearParentCache()', ADMIN_HTML)
        self.assertIn('parent_task_id=', ADMIN_HTML)
        self.assertIn('清除父任务缓存', ADMIN_HTML)

    def test_task_id_can_be_searched_and_copied(self):
        self.assertIn('id="task-search"', ADMIN_HTML)
        self.assertIn('function findTask()', ADMIN_HTML)
        self.assertIn('function copyTaskId()', ADMIN_HTML)
        self.assertIn('/api/admin/tasks/${encodeURIComponent(taskId)}', ADMIN_HTML)
        self.assertIn('x.task_id', ADMIN_HTML)

    def test_parent_task_search_resets_and_refreshes_the_active_gallery(self):
        self.assertIn('async function findTask(){', ADMIN_HTML)
        self.assertIn("const isTaskId = /^[0-9a-f]{32}$/i.test(q);", ADMIN_HTML)
        self.assertIn('if(isTaskId && await showTaskDetail(q)){', ADMIN_HTML)
        self.assertNotIn('if(!q) return;', ADMIN_HTML)
        self.assertIn('galleryPage = 1;', ADMIN_HTML)
        self.assertIn("if(activeTab === 'gallery') loadGalleryPage();", ADMIN_HTML)

    def test_request_pagination_defaults_to_twenty_everywhere(self):
        self.assertIn('<option value="20" selected>20</option>', ADMIN_HTML)
        self.assertIn('let requestPageSize = 20;', ADMIN_HTML)
        self.assertIn('limit: int = Query(20, ge=10, le=200)', APP_SOURCE)

    def test_request_api_returns_page_metadata(self):
        self.assertIn('offset: int = Query(0, ge=0)', APP_SOURCE)
        self.assertIn('"total": total', APP_SOURCE)
        self.assertIn('"has_more": offset + len(rows) < total', APP_SOURCE)

    def test_cached_results_keep_gallery_file_association(self):
        self.assertIn('"output_file": _cached.get("output_file") or _static_name_from_url', APP_SOURCE)
        self.assertIn('"output_file": _db_url.get("output_file") or _static_name_from_url', APP_SOURCE)
        self.assertIn('idx_requests_parent_output', APP_SOURCE)
        self.assertIn('idx_requests_parent_id', APP_SOURCE)
        self.assertIn('output_file IS NOT NULL OR output_url IS NOT NULL', APP_SOURCE)

    def test_task_tracking_schema_and_routes_exist(self):
        self.assertIn('task_id TEXT', APP_SOURCE)
        self.assertIn('response_json TEXT', APP_SOURCE)
        self.assertIn('@app.get("/api/tasks/{task_id}")', APP_SOURCE)
        self.assertIn('@app.get("/api/admin/tasks/{task_id}")', APP_SOURCE)
        self.assertIn('@app.exception_handler(RequestValidationError)', APP_SOURCE)


if __name__ == '__main__':
    unittest.main()
