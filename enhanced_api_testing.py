"""
Assignment 2: Improved API Testing with Class-Specific Images
Generate realistic test images for different classes instead of random noise
"""

import requests
import json
import os
from PIL import Image, ImageDraw
import numpy as np
import io
import matplotlib.pyplot as plt

def create_airplane_image(size=(64, 64)):
    """Create a realistic airplane image"""
    # Sky blue background
    img = Image.new('RGB', size, color=(135, 206, 250))
    draw = ImageDraw.Draw(img)
    
    center_x, center_y = size[0] // 2, size[1] // 2
    
    # Airplane body (white fuselage)
    draw.ellipse([center_x-18, center_y-2, center_x+18, center_y+2], fill='white')
    
    # Wings (light gray)
    draw.ellipse([center_x-22, center_y-6, center_x+22, center_y+6], fill='lightgray')
    
    # Tail fin
    draw.polygon([center_x+15, center_y-1, center_x+25, center_y-8, center_x+25, center_y+8], fill='white')
    
    # Cockpit (darker)
    draw.ellipse([center_x-18, center_y-1, center_x-10, center_y+1], fill='darkblue')
    
    # Engine details
    draw.circle([center_x-8, center_y-4], 2, fill='gray')
    draw.circle([center_x-8, center_y+4], 2, fill='gray')
    
    return img

def create_car_image(size=(64, 64)):
    """Create a realistic car image"""
    # Road/ground background
    img = Image.new('RGB', size, color=(169, 169, 169))  # Gray road
    draw = ImageDraw.Draw(img)
    
    center_x, center_y = size[0] // 2, size[1] // 2 + 5
    
    # Car body (red)
    draw.rectangle([center_x-20, center_y-8, center_x+20, center_y+5], fill='red')
    
    # Car roof
    draw.rectangle([center_x-15, center_y-12, center_x+15, center_y-8], fill=(139, 0, 0))  # Dark red
    
    # Wheels (black circles)
    draw.circle([center_x-12, center_y+8], 6, fill='black')
    draw.circle([center_x+12, center_y+8], 6, fill='black')
    
    # Wheel rims
    draw.circle([center_x-12, center_y+8], 3, fill='gray')
    draw.circle([center_x+12, center_y+8], 3, fill='gray')
    
    # Windows (light blue)
    draw.rectangle([center_x-12, center_y-11, center_x+12, center_y-9], fill='lightblue')
    
    # Headlights
    draw.circle([center_x-18, center_y-3], 2, fill='yellow')
    draw.circle([center_x+18, center_y-3], 2, fill='yellow')
    
    return img

def create_ship_image(size=(64, 64)):
    """Create a realistic ship image"""
    # Ocean background
    img = Image.new('RGB', size, color=(0, 100, 200))  # Ocean blue
    draw = ImageDraw.Draw(img)
    
    center_x, center_y = size[0] // 2, size[1] // 2 + 10
    
    # Ship hull (brown)
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
    
    # Sail (white)
    draw.polygon([
        (center_x+2, center_y-20),
        (center_x+15, center_y-15),
        (center_x+15, center_y-5),
        (center_x+2, center_y-10)
    ], fill='white')
    
    # Ship cabin
    draw.rectangle([center_x-8, center_y-10, center_x+8, center_y-5], fill='white')
    
    return img

def create_bird_image(size=(64, 64)):
    """Create a simple bird image"""
    # Sky background
    img = Image.new('RGB', size, color=(173, 216, 230))  # Light blue sky
    draw = ImageDraw.Draw(img)
    
    center_x, center_y = size[0] // 2, size[1] // 2
    
    # Bird body (brown)
    draw.ellipse([center_x-6, center_y-3, center_x+6, center_y+3], fill='brown')
    
    # Bird head
    draw.circle([center_x-8, center_y-1], 4, fill='brown')
    
    # Beak
    draw.polygon([center_x-12, center_y-1, center_x-15, center_y, center_x-12, center_y+1], fill='orange')
    
    # Wings (spread)
    draw.ellipse([center_x-2, center_y-8, center_x+8, center_y+2], fill=(101, 67, 33))  # Dark brown
    draw.ellipse([center_x-8, center_y-8, center_x+2, center_y+2], fill=(101, 67, 33))  # Dark brown
    
    # Eye
    draw.circle([center_x-8, center_y-2], 1, fill='black')
    
    return img

def create_cat_image(size=(64, 64)):
    """Create a simple cat image"""
    # Indoor background
    img = Image.new('RGB', size, color=(245, 245, 220))  # Beige background
    draw = ImageDraw.Draw(img)
    
    center_x, center_y = size[0] // 2, size[1] // 2 + 5
    
    # Cat body (gray)
    draw.ellipse([center_x-12, center_y-5, center_x+12, center_y+10], fill='gray')
    
    # Cat head
    draw.circle([center_x, center_y-10], 8, fill='gray')
    
    # Cat ears
    draw.polygon([center_x-6, center_y-18, center_x-2, center_y-10, center_x-10, center_y-10], fill='gray')
    draw.polygon([center_x+6, center_y-18, center_x+2, center_y-10, center_x+10, center_y-10], fill='gray')
    
    # Eyes
    draw.circle([center_x-3, center_y-12], 2, fill='green')
    draw.circle([center_x+3, center_y-12], 2, fill='green')
    
    # Nose
    draw.polygon([center_x-1, center_y-8, center_x+1, center_y-8, center_x, center_y-6], fill='pink')
    
    # Tail
    draw.ellipse([center_x+10, center_y-15, center_x+15, center_y+5], fill='gray')
    
    return img

def save_test_images():
    """Create and save test images for all classes"""
    print("🎨 Creating class-specific test images...")
    
    image_creators = {
        'airplane': create_airplane_image,
        'automobile': create_car_image,
        'ship': create_ship_image,
        'bird': create_bird_image,
        'cat': create_cat_image
    }
    
    saved_files = []
    
    for class_name, creator_func in image_creators.items():
        img = creator_func()
        filename = f"test_{class_name}.png"
        img.save(filename)
        saved_files.append(filename)
        print(f"   ✅ Created {filename}")
    
    return saved_files

def test_api_with_class_images(base_url="http://localhost:8000", image_files=None):
    """Test API with class-specific images"""
    print(f"\n🧪 Testing API with Class-Specific Images")
    print("=" * 60)
    
    if image_files is None:
        image_files = save_test_images()
    
    results = []
    
    for image_file in image_files:
        class_name = image_file.replace('test_', '').replace('.png', '')
        
        try:
            print(f"\n🔍 Testing {class_name.upper()} image...")
            
            with open(image_file, 'rb') as f:
                files = {'file': (image_file, f, 'image/png')}
                response = requests.post(f"{base_url}/classify", files=files)
            
            if response.status_code == 200:
                data = response.json()
                predicted_class = data['prediction']['class']
                confidence = data['prediction']['confidence']
                
                is_correct = predicted_class == class_name
                
                print(f"   📊 Result:")
                print(f"      Expected: {class_name}")
                print(f"      Predicted: {predicted_class}")
                print(f"      Confidence: {confidence:.3f}")
                print(f"      Correct: {'✅' if is_correct else '❌'}")
                
                if not is_correct:
                    print(f"      Top 3 predictions:")
                    for i, pred in enumerate(data['top_3_predictions'], 1):
                        print(f"        {i}. {pred['class']}: {pred['confidence']:.3f}")
                
                results.append({
                    'image': class_name,
                    'expected': class_name,
                    'predicted': predicted_class,
                    'confidence': confidence,
                    'correct': is_correct
                })
                
            else:
                print(f"   ❌ API Error: {response.status_code}")
                print(f"      {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error testing {class_name}: {e}")
    
    # Summary
    print(f"\n📊 Test Results Summary:")
    print("=" * 40)
    
    if results:
        correct_count = sum(1 for r in results if r['correct'])
        total_count = len(results)
        accuracy = correct_count / total_count * 100
        
        print(f"   Accuracy: {correct_count}/{total_count} ({accuracy:.1f}%)")
        print(f"   Average Confidence: {np.mean([r['confidence'] for r in results]):.3f}")
        
        print(f"\n   Individual Results:")
        for result in results:
            status = "✅" if result['correct'] else "❌"
            print(f"     {status} {result['image']}: {result['predicted']} ({result['confidence']:.3f})")
    
    return results

def create_custom_test_image(class_name):
    """Create a custom test image for a specific class"""
    creators = {
        'airplane': create_airplane_image,
        'automobile': create_car_image,
        'ship': create_ship_image,
        'bird': create_bird_image,
        'cat': create_cat_image
    }
    
    if class_name in creators:
        img = creators[class_name]()
        
        # Save and return bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img, img_bytes.getvalue()
    else:
        raise ValueError(f"No creator available for class: {class_name}")

def visualize_created_images():
    """Show all created test images"""
    print(f"\n🖼️ Visualizing Created Test Images")
    print("=" * 50)
    
    image_files = save_test_images()
    
    fig, axes = plt.subplots(1, len(image_files), figsize=(15, 3))
    
    for i, image_file in enumerate(image_files):
        img = Image.open(image_file)
        class_name = image_file.replace('test_', '').replace('.png', '')
        
        axes[i].imshow(img)
        axes[i].set_title(f"{class_name.title()}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return image_files

def main():
    """Main function to test improved image generation"""
    print("🎯 Assignment 2: Improved Class-Specific Image Testing")
    print("=" * 70)
    
    # Create and visualize images
    image_files = visualize_created_images()
    
    print(f"\n💡 Key Improvements:")
    print("   • Realistic airplane with wings, body, and cockpit ✈️")
    print("   • Detailed car with wheels, windows, and lights 🚗")
    print("   • Ship with hull, mast, and sail ⛵")
    print("   • Bird with wings, beak, and proper colors 🐦")
    print("   • Cat with ears, eyes, and tail 🐱")
    
    print(f"\n🔧 Usage Instructions:")
    print("   1. Start your API server:")
    print("      uvicorn assignment2_api:app --host 0.0.0.0 --port 8000")
    print("   2. Test with specific class:")
    print("      python -c \"from enhanced_api_testing import *; test_api_with_class_images()\"")
    print("   3. Create custom image:")
    print("      img, bytes = create_custom_test_image('airplane')")
    
    # Clean up
    for file in image_files:
        try:
            os.remove(file)
            print(f"   🗑️ Cleaned up {file}")
        except:
            pass

if __name__ == "__main__":
    main()