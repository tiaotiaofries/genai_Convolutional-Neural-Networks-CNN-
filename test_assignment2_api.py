"""
Assignment 2 API Test Client
Test the FastAPI image classification endpoints
"""

import requests
import json
import os
from PIL import Image, ImageDraw
import numpy as np
import io

def create_realistic_test_image(target_class="airplane", size=(64, 64)):
    """Create a realistic test image instead of random noise"""
    
    if target_class == "airplane":
        # Sky blue background with airplane
        img = Image.new('RGB', size, color=(135, 206, 250))
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = size[0] // 2, size[1] // 2
        
        # Airplane body
        draw.ellipse([center_x-18, center_y-2, center_x+18, center_y+2], fill='white')
        # Wings
        draw.ellipse([center_x-22, center_y-6, center_x+22, center_y+6], fill='lightgray')
        # Tail
        draw.polygon([center_x+15, center_y-1, center_x+25, center_y-8, center_x+25, center_y+8], fill='white')
        # Cockpit
        draw.ellipse([center_x-18, center_y-1, center_x-10, center_y+1], fill='blue')
        
    elif target_class == "automobile":
        # Road background with car
        img = Image.new('RGB', size, color=(169, 169, 169))
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = size[0] // 2, size[1] // 2 + 5
        
        # Car body
        draw.rectangle([center_x-20, center_y-8, center_x+20, center_y+5], fill='red')
        # Wheels
        draw.circle([center_x-12, center_y+8], 6, fill='black')
        draw.circle([center_x+12, center_y+8], 6, fill='black')
        # Windows
        draw.rectangle([center_x-12, center_y-11, center_x+12, center_y-9], fill='lightblue')
        
    elif target_class == "ship":
        # Ocean background with ship
        img = Image.new('RGB', size, color=(0, 100, 200))
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = size[0] // 2, size[1] // 2 + 10
        
        # Ship hull
        hull_points = [
            (center_x-25, center_y),
            (center_x+25, center_y),
            (center_x+20, center_y+10),
            (center_x-20, center_y+10)
        ]
        draw.polygon(hull_points, fill='brown')
        # Ship deck
        draw.rectangle([center_x-20, center_y-5, center_x+20, center_y], fill='gray')
        # Mast
        draw.rectangle([center_x-1, center_y-25, center_x+1, center_y], fill='brown')
        
    else:
        # Default: create a simple pattern
        img = Image.new('RGB', size, color=(128, 128, 128))
        draw = ImageDraw.Draw(img)
        center_x, center_y = size[0] // 2, size[1] // 2
        draw.ellipse([center_x-15, center_y-15, center_x+15, center_y+15], fill=(255, 255, 255))
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()

def create_test_image(class_name="airplane", size=(64, 64)):
    """Create a realistic test image for a specific class"""
    return create_realistic_test_image(class_name, size)

def test_api_endpoints(base_url="http://localhost:8000"):
    """Test all API endpoints"""
    print("🧪 Testing Assignment 2 API Endpoints")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1️⃣ Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Root endpoint
    print("\n2️⃣ Testing Root Endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   API: {data['message']}")
            print(f"   Version: {data['version']}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Model info
    print("\n3️⃣ Testing Model Info...")
    try:
        response = requests.get(f"{base_url}/model/info")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Model: {data['model_name']}")
            print(f"   Classes: {len(data['classes'])}")
            print(f"   Parameters: {data['architecture']['total_parameters']:,}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Classes endpoint
    print("\n4️⃣ Testing Classes Endpoint...")
    try:
        response = requests.get(f"{base_url}/classes")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Total classes: {data['total_classes']}")
            print(f"   Classes: {data['classes']}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 5: Single image classification (improved with realistic airplane)
    print("\n5️⃣ Testing Image Classification with Realistic Airplane...")
    try:
        # Create realistic airplane image instead of random noise
        test_image_data = create_test_image("airplane")
        
        files = {'file': ('realistic_airplane.png', test_image_data, 'image/png')}
        response = requests.post(f"{base_url}/classify", files=files)
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            predicted_class = data['prediction']['class']
            confidence = data['prediction']['confidence']
            
            print(f"   Input: Realistic airplane image")
            print(f"   Prediction: {predicted_class}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Expected: airplane")
            print(f"   Correct: {'✅' if predicted_class == 'airplane' else '❌'}")
            print(f"   Top 3 predictions:")
            for i, pred in enumerate(data['top_3_predictions'], 1):
                print(f"     {i}. {pred['class']}: {pred['confidence']:.3f}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 6: Batch classification with different classes
    print("\n6️⃣ Testing Batch Classification with Different Classes...")
    try:
        # Create multiple test images for different classes
        test_classes = ['airplane', 'automobile', 'ship']
        files = []
        for i, class_name in enumerate(test_classes):
            test_image_data = create_test_image(class_name)
            files.append(('files', (f'test_{class_name}.png', test_image_data, 'image/png')))
        
        response = requests.post(f"{base_url}/classify/batch", files=files)
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Batch size: {data['batch_size']}")
            for i, result in enumerate(data['results']):
                expected_class = test_classes[i]
                if 'prediction' in result:
                    predicted_class = result['prediction']['class']
                    confidence = result['prediction']['confidence']
                    is_correct = predicted_class == expected_class
                    status = "✅" if is_correct else "❌"
                    
                    print(f"   {result['filename']}: Expected {expected_class}")
                    print(f"      → Predicted: {predicted_class} ({confidence:.3f}) {status}")
                else:
                    print(f"   {result['filename']}: Error - {result['error']}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print(f"\n✅ API Testing Complete!")

def test_api_locally():
    """Test API running locally"""
    print("🧪 Testing Local API (make sure server is running)")
    test_api_endpoints("http://localhost:8000")

if __name__ == "__main__":
    test_api_locally()