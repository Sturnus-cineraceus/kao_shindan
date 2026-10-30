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
        assert any(x.get('name') == 'image' and x.get('type') == 'file' and x.get('accept') == 'image/jpeg,image/png' and 'required' in x for x in inputs)
        assert any(tag == 'select' and attrs.get('name') == 'reference_group' and 'required' in attrs for tag, attrs in parser.elements)
        assert {attrs.get('value') for tag, attrs in parser.elements if tag == 'option'} >= {'young_male', 'young_female', 'older_male', 'older_female'}
        assert any(tag == 'button' and attrs.get('type') == 'submit' for tag, attrs in parser.elements)
        assert client.get('/static/app.js').status_code == 200
        assert client.get('/static/style.css').status_code == 200
