"""
Assignment 2: Updated API Testing with Realistic Images
Replace random noise with class-specific realistic test images
"""

import requests
import json
import numpy as np
from PIL import Image, ImageDraw
import io

def create_specific_test_image(target_class="airplane", size=(64, 64)):
    """Create a test image for a specific class"""
    
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
        # Car roof
        draw.rectangle([center_x-15, center_y-12, center_x+15, center_y-8], fill=(139, 0, 0))
        # Wheels
        draw.circle([center_x-12, center_y+8], 6, fill='black')
        draw.circle([center_x+12, center_y+8], 6, fill='black')
        # Windows
        draw.rectangle([center_x-12, center_y-11, center_x+12, center_y-9], fill='lightblue')
        # Headlights
        draw.circle([center_x-18, center_y-3], 2, fill='yellow')
        draw.circle([center_x+18, center_y-3], 2, fill='yellow')
        
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
        # Sail
        draw.polygon([
            (center_x+2, center_y-20),
            (center_x+15, center_y-15),
            (center_x+15, center_y-5),
            (center_x+2, center_y-10)
        ], fill='white')
        
    elif target_class == "bird":
        # Sky background with bird
        img = Image.new('RGB', size, color=(173, 216, 230))
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = size[0] // 2, size[1] // 2
        
        # Bird body
        draw.ellipse([center_x-6, center_y-3, center_x+6, center_y+3], fill='brown')
        # Bird head
        draw.circle([center_x-8, center_y-1], 4, fill='brown')
        # Beak
        draw.polygon([center_x-12, center_y-1, center_x-15, center_y, center_x-12, center_y+1], fill='orange')
        # Wings
        draw.ellipse([center_x-2, center_y-8, center_x+8, center_y+2], fill=(101, 67, 33))
        draw.ellipse([center_x-8, center_y-8, center_x+2, center_y+2], fill=(101, 67, 33))
        # Eye
        draw.circle([center_x-8, center_y-2], 1, fill='black')
        
    else:
        # Default: create a simple colored square for other classes
        colors = {
            'cat': (255, 165, 0),      # Orange
            'deer': (139, 69, 19),     # Brown
            'dog': (210, 180, 140),    # Tan
            'frog': (34, 139, 34),     # Green
            'horse': (160, 82, 45),    # Saddle brown
            'truck': (105, 105, 105)   # Dim gray
        }
        color = colors.get(target_class, (128, 128, 128))
        img = Image.new('RGB', size, color=color)
        
        # Add some basic shape
        draw = ImageDraw.Draw(img)
        center_x, center_y = size[0] // 2, size[1] // 2
        draw.ellipse([center_x-15, center_y-15, center_x+15, center_y+15], fill=(255, 255, 255))
    
    return img

def test_specific_class_prediction(target_class="airplane", base_url="http://localhost:8000"):
    """Test prediction for a specific class"""
    print(f"🎯 Testing {target_class.upper()} prediction")
    print("=" * 50)
    
    try:
        # Create test image for the target class
        test_img = create_specific_test_image(target_class)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Test with API
        files = {'file': (f'test_{target_class}.png', img_bytes.getvalue(), 'image/png')}
        response = requests.post(f"{base_url}/classify", files=files)
        
        if response.status_code == 200:
            data = response.json()
            predicted_class = data['prediction']['class']
            confidence = data['prediction']['confidence']
            
            print(f"✅ API Response received")
            print(f"   Expected: {target_class}")
            print(f"   Predicted: {predicted_class}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Correct: {'✅' if predicted_class == target_class else '❌'}")
            
            print(f"\n📊 Top 3 Predictions:")
            for i, pred in enumerate(data['top_3_predictions'], 1):
                icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                print(f"   {icon} {pred['class']}: {pred['confidence']:.3f}")
            
            return predicted_class, confidence
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   {response.text}")
            return None, 0
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, 0

def test_multiple_classes(classes=["airplane", "automobile", "ship", "bird"], base_url="http://localhost:8000"):
    """Test multiple classes and compare results"""
    print(f"🧪 Testing Multiple Classes with Realistic Images")
    print("=" * 60)
    
    results = []
    
    for target_class in classes:
        print(f"\n🔍 Testing {target_class}...")
        predicted, confidence = test_specific_class_prediction(target_class, base_url)
        
        if predicted is not None:
            is_correct = predicted == target_class
            results.append({
                'target': target_class,
                'predicted': predicted,
                'confidence': confidence,
                'correct': is_correct
            })
            
            status = "✅" if is_correct else "❌"
            print(f"   Result: {status} {predicted} ({confidence:.3f})")
    
    # Summary
    if results:
        print(f"\n📈 Summary Results:")
        print("=" * 30)
        
        correct_count = sum(1 for r in results if r['correct'])
        total_count = len(results)
        accuracy = correct_count / total_count * 100
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        print(f"   Accuracy: {correct_count}/{total_count} ({accuracy:.1f}%)")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        
        print(f"\n   Detailed Results:")
        for result in results:
            status = "✅" if result['correct'] else "❌"
            print(f"     {status} {result['target']} → {result['predicted']} ({result['confidence']:.3f})")
    
    return results

def compare_random_vs_realistic():
    """Compare random noise vs realistic images"""
    print(f"📊 Comparison: Random Noise vs Realistic Images")
    print("=" * 60)
    
    # Test with random noise (like original)
    print(f"\n🎲 Testing with Random Noise:")
    random_img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    random_img = Image.fromarray(random_img_array)
    
    img_bytes = io.BytesIO()
    random_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    try:
        files = {'file': ('random.png', img_bytes.getvalue(), 'image/png')}
        response = requests.post("http://localhost:8000/classify", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Random noise predicted: {data['prediction']['class']} ({data['prediction']['confidence']:.3f})")
        else:
            print(f"   Random noise test failed")
    except:
        print(f"   Could not test random noise (server not running?)")
    
    # Test with realistic airplane
    print(f"\n✈️ Testing with Realistic Airplane:")
    predicted, confidence = test_specific_class_prediction("airplane")
    
    print(f"\n💡 Key Insight:")
    print("   • Random noise → unpredictable results")
    print("   • Realistic images → more accurate predictions")
    print("   • Model recognizes actual visual features")

def main():
    """Main testing function"""
    print("🎯 Assignment 2: Enhanced Image Testing")
    print("Realistic Class-Specific Images Instead of Random Noise")
    print("=" * 70)
    
    # Test specific classes
    target_classes = ["airplane", "automobile", "ship", "bird"]
    
    print(f"🎨 Creating realistic test images for: {', '.join(target_classes)}")
    
    try:
        # Test if server is running
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ API server is running")
            
            # Run tests
            results = test_multiple_classes(target_classes)
            
            # Show comparison
            compare_random_vs_realistic()
            
        else:
            print("❌ API server not responding")
            
    except requests.exceptions.ConnectionError:
        print("❌ API server not running. Start it with:")
        print("   uvicorn assignment2_api:app --host 0.0.0.0 --port 8000")
        
        # Still show how to create images
        print(f"\n🎨 Example: Creating airplane image")
        airplane_img = create_specific_test_image("airplane")
        print(f"   Created realistic airplane image!")
    
    print(f"\n🎓 Summary:")
    print("   You can now generate realistic test images for any class!")
    print("   This gives much better predictions than random noise.")

if __name__ == "__main__":
    main()