# Assignment 2: Convolutional Neural Networks (CNN)

## CNN Architecture Specification

The CNN follows the exact architecture specified in the assignment:

```
Input: RGB image of size 64×64×3
├── Conv2D (16 filters, 3×3 kernel, stride=1, padding=1)
├── ReLU activation
├── MaxPooling2D (2×2 kernel, stride=2)
├── Conv2D (32 filters, 3×3 kernel, stride=1, padding=1)
├── ReLU activation
├── MaxPooling2D (2×2 kernel, stride=2)
├── Flatten
├── Fully Connected (100 units)
├── ReLU activation
└── Fully Connected (10 units - output classes)
```

**Model Statistics:**
- Total Parameters: 825,398
- Model Size: ~3.15 MB
- Training Dataset: CIFAR-10 (resized to 64×64)
- Test Accuracy: 65.68%

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd assignment2

# Install dependencies
pip install -r requirements_assignment2.txt
```

### 2. Run the CNN Training (Optional)

```bash
# Train the CNN model from scratch
python assignment2_cnn_classifier.py
```

### 3. Start the API Server

```bash
# Start the FastAPI server
uvicorn assignment2_api:app --host 0.0.0.0 --port 8000
```

### 4. Test the API

```bash
# Run the test client
python test_assignment2_api.py

# Or test manually with curl
curl -X GET http://localhost:8000/health
```

## Docker Deployment

### Build the Docker Image

```bash
docker build -f Dockerfile.assignment2 -t assignment2-cnn .
```

### Run the Container

```bash
docker run -p 8000:8000 assignment2-cnn
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Model Performance

**Training Results:**
- Dataset: CIFAR-10 (50,000 training, 10,000 test images)
- Image Size: Resized from 32×32 to 64×64 RGB
- Training Epochs: 10
- Final Test Accuracy: 65.68%
- Training Time: ~6 minutes on CPU

**Classes Supported:**
```
0: airplane    5: dog
1: automobile  6: frog
2: bird        7: horse
3: cat         8: ship
4: deer        9: truck
```

## Technical Details

### Architecture Implementation

The CNN is implemented in PyTorch following the exact specifications:

```python
class AssignmentCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AssignmentCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 16 * 32, 100)
        self.fc2 = nn.Linear(100, num_classes)
```

## Testing

### Run All Tests

```bash
# Test the model inference
python assignment2_demo.py

# Test the API endpoints
python test_assignment2_api.py
```

### Manual Testing

```bash
# Health check
curl http://localhost:8000/health

# Model information
curl http://localhost:8000/model/info

# Classes list
curl http://localhost:8000/classes
```
