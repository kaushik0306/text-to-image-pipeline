import pytest
import json
import sys
sys.path.insert(0, '.')
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_generate_valid(client):
    """Test the /generate endpoint with a valid prompt"""
    response = client.post('/generate', data=json.dumps({
        "prompt": "A sports bike in a cricket stadium"
    }), content_type='application/json')
    
    assert response.status_code == 200
    data = response.get_json()
    assert 'image_path' in data
    assert data['message'] == "Image generated successfully!"

def test_generate_invalid(client):
    """Test the /generate endpoint with an invalid prompt"""
    response = client.post('/generate', data=json.dumps({
        "prompt": 12345  # Invalid prompt, should be a string
    }), content_type='application/json')
    
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data

def test_analyze_valid(client):
    """Test the /analyze endpoint with a valid image path"""
    response = client.post('/analyze', data=json.dumps({
        "image_path": "generated_image.png"  # Assume the file exists
    }), content_type='application/json')
    
    assert response.status_code == 200
    data = response.get_json()
    assert 'overall_probabilities' in data
    assert 'region_specific_analysis' in data


def test_analyze_no_image_path(client):
    """Test the /analyze endpoint without providing an image path"""
    response = client.post('/analyze', data=json.dumps({}), content_type='application/json')
    
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data

