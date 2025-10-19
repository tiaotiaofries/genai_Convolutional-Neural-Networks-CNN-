"""
Assignment 2: Enhanced Image Generation and Class-Specific Testing
Improving image generation to create more realistic class-specific images
"""

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import io
from assignment2_cnn_classifier import AssignmentCNN
from helper_lib import load_model, get_device

def load_real_cifar10_samples():
    """Load real CIFAR-10 samples for each class"""
    print("📊 Loading real CIFAR-10 samples...")
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to 64x64 as needed
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform
    )
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Get one sample from each class
    class_samples = {}
    class_counts = {i: 0 for i in range(10)}
    
    for image, label in test_dataset:
        if class_counts[label] == 0:  # Get first sample of each class
            class_samples[label] = image
            class_counts[label] = 1
        
        # Stop when we have one sample from each class
        if all(count > 0 for count in class_counts.values()):
            break
    
    return class_samples, class_names

def create_synthetic_airplane_image(size=(64, 64)):
    """Create a simple synthetic airplane-like image"""
    # Create a simple airplane shape
    img = Image.new('RGB', size, color='lightblue')  # Sky background
    draw = ImageDraw.Draw(img)
    
    # Draw airplane body (fuselage)
    center_x, center_y = size[0] // 2, size[1] // 2
    draw.ellipse([center_x-20, center_y-3, center_x+20, center_y+3], fill='gray')
    
    # Draw wings
    draw.ellipse([center_x-15, center_y-8, center_x+15, center_y+8], fill='darkgray')
    
    # Draw tail
    draw.polygon([center_x+15, center_y-2, center_x+25, center_y-8, center_x+25, center_y+8], fill='gray')
    
    return img

def create_synthetic_class_images():
    """Create synthetic images for different classes"""
    synthetic_images = {}
    
    # Airplane
    airplane_img = create_synthetic_airplane_image()
    synthetic_images['airplane'] = airplane_img
    
    # Car (simple rectangle)
    car_img = Image.new('RGB', (64, 64), color='lightgray')
    draw = ImageDraw.Draw(car_img)
    draw.rectangle([10, 30, 54, 45], fill='red')  # Car body
    draw.circle([20, 45], 5, fill='black')  # Wheel 1
    draw.circle([44, 45], 5, fill='black')  # Wheel 2
    synthetic_images['automobile'] = car_img
    
    # Ship (simple boat shape)
    ship_img = Image.new('RGB', (64, 64), color='lightblue')
    draw = ImageDraw.Draw(ship_img)
    draw.polygon([10, 40, 54, 40, 50, 50, 14, 50], fill='brown')  # Hull
    draw.rectangle([30, 20, 34, 40], fill='gray')  # Mast
    synthetic_images['ship'] = ship_img
    
    return synthetic_images

def preprocess_pil_image(pil_image):
    """Convert PIL image to tensor format for model"""
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    tensor_image = transform(pil_image).unsqueeze(0)  # Add batch dimension
    return tensor_image

def test_model_with_specific_classes(model, device, target_classes=['airplane', 'automobile', 'ship']):
    """Test model with real and synthetic images of specific classes"""
    print(f"\n🎯 Testing Model with Specific Classes: {target_classes}")
    print("=" * 60)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Load real samples
    try:
        real_samples, _ = load_real_cifar10_samples()
        print("✅ Real CIFAR-10 samples loaded")
    except:
        print("⚠️ Could not load real samples, using synthetic only")
        real_samples = {}
    
    # Create synthetic samples
    synthetic_samples = create_synthetic_class_images()
    print("✅ Synthetic samples created")
    
    model.eval()
    results = []
    
    for target_class in target_classes:
        target_idx = class_names.index(target_class)
        
        print(f"\n🔍 Testing {target_class.upper()}:")
        
        # Test with real image if available
        if target_idx in real_samples:
            real_image = real_samples[target_idx].unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(real_image)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = class_names[predicted.item()]
                is_correct = predicted_class == target_class
                
                print(f"   📸 Real {target_class}:")
                print(f"      Predicted: {predicted_class} ({confidence.item():.3f})")
                print(f"      Correct: {'✅' if is_correct else '❌'}")
                
                results.append({
                    'type': 'real',
                    'target': target_class,
                    'predicted': predicted_class,
                    'confidence': confidence.item(),
                    'correct': is_correct
                })
        
        # Test with synthetic image if available
        if target_class in synthetic_samples:
            synthetic_pil = synthetic_samples[target_class]
            synthetic_tensor = preprocess_pil_image(synthetic_pil).to(device)
            
            with torch.no_grad():
                outputs = model(synthetic_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = class_names[predicted.item()]
                is_correct = predicted_class == target_class
                
                print(f"   🎨 Synthetic {target_class}:")
                print(f"      Predicted: {predicted_class} ({confidence.item():.3f})")
                print(f"      Correct: {'✅' if is_correct else '❌'}")
                
                results.append({
                    'type': 'synthetic',
                    'target': target_class,
                    'predicted': predicted_class,
                    'confidence': confidence.item(),
                    'correct': is_correct
                })
    
    return results, synthetic_samples

def visualize_class_specific_results(model, device, synthetic_samples, class_names):
    """Visualize results for different image types"""
    print(f"\n🖼️ Visualizing Class-Specific Results")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Test different image types
    test_cases = [
        ('Random Noise', torch.randn(1, 3, 64, 64)),
        ('Blue Sky', torch.ones(1, 3, 64, 64) * 0.5),  # Normalized blue
        ('Gray Pattern', torch.zeros(1, 3, 64, 64))
    ]
    
    for i, (case_name, test_image) in enumerate(test_cases):
        test_image = test_image.to(device)
        
        with torch.no_grad():
            outputs = model(test_image)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = class_names[predicted.item()]
        
        # Convert image for display
        display_img = test_image.squeeze().cpu()
        display_img = display_img * 0.5 + 0.5  # Denormalize
        display_img = display_img.permute(1, 2, 0)
        
        axes[0, i].imshow(display_img)
        axes[0, i].set_title(f"{case_name}\nPred: {predicted_class}\nConf: {confidence.item():.3f}")
        axes[0, i].axis('off')
    
    # Show synthetic images
    synthetic_names = list(synthetic_samples.keys())
    for i, name in enumerate(synthetic_names[:3]):
        if i < 3:
            axes[1, i].imshow(synthetic_samples[name])
            axes[1, i].set_title(f"Synthetic {name.title()}")
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def create_class_targeted_test_images():
    """Create test images that should be more recognizable as specific classes"""
    print(f"\n🎨 Creating Class-Targeted Test Images")
    print("=" * 50)
    
    # Instead of random noise, create images with class-specific patterns
    class_patterns = {}
    
    # Airplane pattern - horizontal lines (wings)
    airplane_pattern = torch.zeros(3, 64, 64)
    airplane_pattern[:, 30:34, :] = 0.8  # Horizontal wing line
    airplane_pattern[:, 32, 20:44] = 1.0  # Fuselage
    class_patterns['airplane'] = airplane_pattern
    
    # Car pattern - rectangular shape
    car_pattern = torch.zeros(3, 64, 64)
    car_pattern[:, 25:40, 15:50] = 0.7  # Car body
    car_pattern[:, 40:45, 20:25] = 0.3  # Wheel 1
    car_pattern[:, 40:45, 40:45] = 0.3  # Wheel 2
    class_patterns['automobile'] = car_pattern
    
    # Ship pattern - boat-like shape
    ship_pattern = torch.zeros(3, 64, 64)
    ship_pattern[:, 35:45, 10:54] = 0.6  # Hull
    ship_pattern[:, 20:35, 30:34] = 0.8  # Mast
    class_patterns['ship'] = ship_pattern
    
    return class_patterns

def test_improved_image_generation():
    """Test the model with improved, class-specific image generation"""
    print("🚀 Assignment 2: Enhanced Class-Specific Image Testing")
    print("=" * 60)
    
    # Load model
    device = get_device()
    model = AssignmentCNN(num_classes=10).to(device)
    model = load_model(model, 'models/assignment2_cnn.pth')
    model.eval()
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("✅ Model loaded successfully!")
    
    # Test with specific classes
    target_classes = ['airplane', 'automobile', 'ship']
    results, synthetic_samples = test_model_with_specific_classes(
        model, device, target_classes
    )
    
    # Create and test pattern-based images
    print(f"\n🎯 Testing Pattern-Based Images")
    print("=" * 40)
    
    class_patterns = create_class_targeted_test_images()
    
    for class_name, pattern in class_patterns.items():
        pattern_tensor = pattern.unsqueeze(0).to(device)  # Add batch dimension
        
        with torch.no_grad():
            outputs = model(pattern_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = class_names[predicted.item()]
            is_correct = predicted_class == class_name
            
            print(f"   🎨 Pattern {class_name}:")
            print(f"      Predicted: {predicted_class} ({confidence.item():.3f})")
            print(f"      Correct: {'✅' if is_correct else '❌'}")
    
    # Visualize results
    visualize_class_specific_results(model, device, synthetic_samples, class_names)
    
    # Summary
    print(f"\n📊 Summary of Results:")
    correct_real = sum(1 for r in results if r['type'] == 'real' and r['correct'])
    total_real = sum(1 for r in results if r['type'] == 'real')
    correct_synthetic = sum(1 for r in results if r['type'] == 'synthetic' and r['correct'])
    total_synthetic = sum(1 for r in results if r['type'] == 'synthetic')
    
    print(f"   Real images: {correct_real}/{total_real} correct")
    print(f"   Synthetic images: {correct_synthetic}/{total_synthetic} correct")
    
    print(f"\n💡 Key Insights:")
    print("   • Random noise → unpredictable predictions")
    print("   • Real CIFAR-10 → best performance")
    print("   • Synthetic patterns → can guide predictions")
    print("   • Class-specific features matter for recognition")

def create_airplane_focused_api_test():
    """Create a test specifically for airplane images"""
    print(f"\n✈️ Creating Airplane-Focused API Test")
    print("=" * 50)
    
    # Create a better airplane image
    airplane_img = Image.new('RGB', (64, 64), color=(135, 206, 250))  # Sky blue
    draw = ImageDraw.Draw(airplane_img)
    
    # More detailed airplane
    # Fuselage
    draw.ellipse([20, 30, 44, 34], fill='white')
    # Wings
    draw.ellipse([15, 28, 49, 36], fill='lightgray')
    # Tail
    draw.polygon([42, 30, 50, 24, 50, 40], fill='white')
    # Nose
    draw.polygon([20, 31, 15, 32, 20, 33], fill='gray')
    
    # Save as bytes for API testing
    img_bytes = io.BytesIO()
    airplane_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return airplane_img, img_bytes.getvalue()

if __name__ == "__main__":
    test_improved_image_generation()
    
    # Create airplane test image
    airplane_pil, airplane_bytes = create_airplane_focused_api_test()
    
    print(f"\n✈️ Airplane test image created!")
    print("   Use this with your API:")
    print("   • Save the airplane image")
    print("   • Test with: curl -X POST http://localhost:8000/classify -F 'file=@airplane.png'")
    print("   • Should predict 'airplane' with higher confidence!")