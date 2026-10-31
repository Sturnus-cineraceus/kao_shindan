from html.parser import HTMLParser

from fastapi.testclient import TestClient

from app.main import app


class FormParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


def test_index_form_and_assets():
    with TestClient(app) as client:
        response = client.get('/')
        assert response.status_code == 200
        assert response.headers['content-type'].startswith('text/html')
        parser = FormParser()
        parser.feed(response.text)
        inputs = [attrs for tag, attrs in parser.elements if tag == 'input']
        checkboxes = [x for x in inputs if x.get('type') == 'checkbox']
        assert len(checkboxes) == 1
        assert checkboxes[0].get('name') == 'additional_feature' and checkboxes[0].get('value') == 'face_aspect_ratio'
        assert all(any(attrs.get('id') == element for _, attrs in parser.elements) for element in ('result-profile', 'result-dimension'))
        assert any(x.get('name') == 'image' and x.get('type') == 'file' and x.get('accept') == 'image/jpeg,image/png' and 'required' in x for x in inputs)
        assert any(tag == 'div' and attrs.get('id') == 'image-preview' and 'hidden' in attrs for tag, attrs in parser.elements)
        assert any(tag == 'img' and attrs.get('id') == 'image-preview-image' for tag, attrs in parser.elements)
        assert any(tag == 'select' and attrs.get('name') == 'reference_group' and 'required' in attrs for tag, attrs in parser.elements)
        assert {attrs.get('value') for tag, attrs in parser.elements if tag == 'option'} >= {'young_male', 'young_female', 'older_male', 'older_female'}
        assert any(tag == 'button' and attrs.get('type') == 'submit' for tag, attrs in parser.elements)
        assert client.get('/static/app.js').status_code == 200
        assert all(label in response.text for label in ('美形スコア', '平均顔距離', 'ボリュームゾーン度'))
        assert all(any(attrs.get('id') == element for _, attrs in parser.elements) for element in ('score', 'distance', 'volume-zone-score'))
        script = client.get('/static/app.js').text
        assert "input.addEventListener('change'" in script
        assert 'featureRevision++' in script and 'submittedRevision !== featureRevision' in script
        assert 'result.hidden = true' in script and 'destroyVolumeZoneChart();' in script
        assert "data.volume_zone_score" in script and "data.distance" in script
        assert client.get('/static/style.css').status_code == 200


def test_volume_zone_chart_assets_and_contract():
    with TestClient(app) as client:
        html = client.get('/').text
        parser = FormParser()
        parser.feed(html)
        assert any(tag == 'canvas' and attrs.get('id') == 'volume-zone-chart'
                   and attrs.get('role') == 'img' and attrs.get('aria-label')
                   for tag, attrs in parser.elements)
        scripts = [attrs.get('src', '') for tag, attrs in parser.elements if tag == 'script']
        assert scripts.index('/static/vendor/chart.umd.min.js') < scripts.index('/static/app.js')
        assert all(src.startswith('/static/') for src in scripts)
        vendor = client.get('/static/vendor/chart.umd.min.js')
        assert vendor.status_code == 200
        assert 'Chart.js v4.6.1' in vendor.text[:200]
        assert client.get('/static/vendor/Chart.js.LICENSE.md').status_code == 200
        script = client.get('/static/app.js').text
        assert 'data.volume_zone_distribution.points.map' in script
        assert 'x: point.distance, y: point.relative_density' in script
        assert 'data.volume_zone_distribution.mode_distance' in script
        assert '{ x: data.distance, y: data.volume_zone_score }' in script
        assert all(label in script for label in ('距離分布', 'この顔', '最頻距離'))
        assert 'volumeZoneChart.destroy()' in script
        assert 'Math.sqrt' not in script and 'chi' not in script.lower()
        for field in ('score', 'distance', 'volume-zone-score'):
            assert f"document.querySelector('#{field}').textContent" in script
