# Assignment 2: Convolutional Neural Networks (CNN)# Assignment 2: Convolutional Neural Networks (CNN) # Assignment 2: CNN Image Classification API



## CNN Architecture Specification



The CNN follows the exact architecture specified in the assignment:## CNN Architecture SpecificationThis repository contains the implementation of Assignment 2 for the Applied Generative AI course. The assignment involves implementing a Convolutional Neural Network (CNN) with a specific architecture and deploying it as a FastAPI service with Docker containerization.



```

Input: RGB image of size 64×64×3

├── Conv2D (16 filters, 3×3 kernel, stride=1, padding=1)The CNN follows the exact architecture specified in the assignment:## 🎯 Assignment Overview

├── ReLU activation

├── MaxPooling2D (2×2 kernel, stride=2)

├── Conv2D (32 filters, 3×3 kernel, stride=1, padding=1)

├── ReLU activation```**Assignment 2** consists of three main parts:

├── MaxPooling2D (2×2 kernel, stride=2)

├── FlattenInput: RGB image of size 64×64×31. **CNN Architecture Implementation**: Implement a CNN matching the exact specifications

├── Fully Connected (100 units)

├── ReLU activation├── Conv2D (16 filters, 3×3 kernel, stride=1, padding=1)2. **FastAPI Integration**: Create API endpoints for image classification

└── Fully Connected (10 units - output classes)

```├── ReLU activation3. **Docker Deployment**: Containerize the application for production deployment



**Model Statistics:**├── MaxPooling2D (2×2 kernel, stride=2)

- Total Parameters: 825,398

- Model Size: ~3.15 MB├── Conv2D (32 filters, 3×3 kernel, stride=1, padding=1)## 🏗️ CNN Architecture Specification

- Training Dataset: CIFAR-10 (resized to 64×64)

- Test Accuracy: 65.68%├── ReLU activation



### 1. Setup Environment├── MaxPooling2D (2×2 kernel, stride=2)The CNN follows the exact architecture specified in the assignment:



```bash├── Flatten

# Clone the repository

git clone <repository-url>├── Fully Connected (100 units)```

cd assignment2

├── ReLU activationInput: RGB image of size 64×64×3

# Install dependencies

pip install -r requirements_assignment2.txt└── Fully Connected (10 units - output classes)├── Conv2D (16 filters, 3×3 kernel, stride=1, padding=1)

```

```├── ReLU activation

### 2. Run the CNN Training (Optional)

├── MaxPooling2D (2×2 kernel, stride=2)

```bash

# Train the CNN model from scratch**Model Statistics:**├── Conv2D (32 filters, 3×3 kernel, stride=1, padding=1)

python assignment2_cnn_classifier.py

```- Total Parameters: 825,398├── ReLU activation



### 3. Start the API Server- Model Size: ~3.15 MB├── MaxPooling2D (2×2 kernel, stride=2)



```bash- Training Dataset: CIFAR-10 (resized to 64×64)├── Flatten

# Start the FastAPI server

uvicorn assignment2_api:app --host 0.0.0.0 --port 8000- Test Accuracy: 65.68%├── Fully Connected (100 units)

```

├── ReLU activation

### 4. Test the API

└── Fully Connected (10 units - output classes)

```bash

# Run the test client### 1. Setup Environment```

python test_assignment2_api.py



# Or test manually with curl

curl -X GET http://localhost:8000/health```bash**Model Statistics:**

```

# Clone the repository- Total Parameters: 825,398

## Docker Deployment

git clone <repository-url>- Model Size: ~3.15 MB

### Build the Docker Image

cd assignment2- Training Dataset: CIFAR-10 (resized to 64×64)

```bash

docker build -f Dockerfile.assignment2 -t assignment2-cnn .- Test Accuracy: 65.68%

```

# Install dependencies

### Run the Container

pip install -r requirements_assignment2.txt## 📁 Repository Structure

```bash

docker run -p 8000:8000 assignment2-cnn```

```

```

### Health Check

### 2. Run the CNN Training (Optional)assignment2/

```bash

curl http://localhost:8000/health├── assignment2_cnn_classifier.py    # CNN implementation and training

```

```bash├── assignment2_api.py               # FastAPI server implementation

## Model Performance

# Train the CNN model from scratch├── assignment2_demo.py              # Complete demonstration

**Training Results:**

- Dataset: CIFAR-10 (50,000 training, 10,000 test images)python assignment2_cnn_classifier.py├── test_assignment2_api.py          # API testing client

- Image Size: Resized from 32×32 to 64×64 RGB

- Training Epochs: 10```├── Dockerfile.assignment2          # Docker configuration

- Final Test Accuracy: 65.68%

- Training Time: ~6 minutes on CPU├── requirements_assignment2.txt     # Python dependencies



**Classes Supported:**### 3. Start the API Server├── models/

```

0: airplane    5: dog│   └── assignment2_cnn.pth         # Trained model weights

1: automobile  6: frog

2: bird        7: horse```bash└── README.md                        # This file

3: cat         8: ship

4: deer        9: truck# Start the FastAPI server```

```

uvicorn assignment2_api:app --host 0.0.0.0 --port 8000

## Technical Details

```## 🚀 Quick Start

### Architecture Implementation



The CNN is implemented in PyTorch following the exact specifications:

### 4. Test the API### 1. Setup Environment

```python

class AssignmentCNN(nn.Module):

    def __init__(self, num_classes=10):

        super(AssignmentCNN, self).__init__()```bash```bash

        

        # First convolutional block# Run the test client# Clone the repository

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)python test_assignment2_api.pygit clone <repository-url>

        

        # Second convolutional block  cd assignment2

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)# Or test manually with curl

        

        # Fully connected layerscurl -X GET http://localhost:8000/health# Install dependencies

        self.fc1 = nn.Linear(16 * 16 * 32, 100)

        self.fc2 = nn.Linear(100, num_classes)```pip install -r requirements_assignment2.txt

```

```

## Testing



### Run All Tests

## Docker Deployment### 2. Run the CNN Training (Optional)

```bash

# Test the model inference

python assignment2_demo.py

### Build the Docker Image```bash

# Test the API endpoints

python test_assignment2_api.py# Train the CNN model from scratch

```

```bashpython assignment2_cnn_classifier.py

### Manual Testing

docker build -f Dockerfile.assignment2 -t assignment2-cnn .```

```bash

# Health check```

curl http://localhost:8000/health

### 3. Start the API Server

# Model information

curl http://localhost:8000/model/info### Run the Container



# Classes list```bash

curl http://localhost:8000/classes

``````bash# Start the FastAPI server

docker run -p 8000:8000 assignment2-cnnuvicorn assignment2_api:app --host 0.0.0.0 --port 8000

``````



### Health Check### 4. Test the API



```bash```bash

curl http://localhost:8000/health# Run the test client

```python test_assignment2_api.py



## Model Performance# Or test manually with curl

curl -X GET http://localhost:8000/health

**Training Results:**```

- Dataset: CIFAR-10 (50,000 training, 10,000 test images)

- Image Size: Resized from 32×32 to 64×64 RGB## 🌐 API Endpoints

- Training Epochs: 10

- Final Test Accuracy: 65.68%| Endpoint | Method | Description |

- Training Time: ~6 minutes on CPU|----------|---------|-------------|

| `/` | GET | API information and documentation |

**Classes Supported:**| `/health` | GET | Health check endpoint |

```| `/model/info` | GET | Model architecture information |

0: airplane    5: dog| `/classes` | GET | List of supported classes |

1: automobile  6: frog| `/classify` | POST | Single image classification |

2: bird        7: horse| `/classify/batch` | POST | Batch image classification |

3: cat         8: ship

4: deer        9: truck### Example API Usage

```

**Single Image Classification:**

## Technical Details```bash

curl -X POST "http://localhost:8000/classify" \

### Architecture Implementation     -H "accept: application/json" \

     -H "Content-Type: multipart/form-data" \

The CNN is implemented in PyTorch following the exact specifications:     -F "file=@image.jpg"

```

```python

class AssignmentCNN(nn.Module):**Response:**

    def __init__(self, num_classes=10):```json

        super(AssignmentCNN, self).__init__(){

          "filename": "image.jpg",

        # First convolutional block  "prediction": {

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)    "class": "airplane",

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)    "class_index": 0,

            "confidence": 0.892

        # Second convolutional block    },

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  "top_3_predictions": [

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)    {"class": "airplane", "class_index": 0, "confidence": 0.892},

            {"class": "bird", "class_index": 2, "confidence": 0.087},

        # Fully connected layers    {"class": "ship", "class_index": 8, "confidence": 0.021}

        self.fc1 = nn.Linear(16 * 16 * 32, 100)  ]

        self.fc2 = nn.Linear(100, num_classes)}

``````



## Testing## 🐳 Docker Deployment



### Run All Tests### Build the Docker Image



```bash```bash

# Test the model inferencedocker build -f Dockerfile.assignment2 -t assignment2-cnn .

python assignment2_demo.py```



# Test the API endpoints### Run the Container

python test_assignment2_api.py

``````bash

docker run -p 8000:8000 assignment2-cnn

### Manual Testing```



```bash### Health Check

# Health check

curl http://localhost:8000/health```bash

curl http://localhost:8000/health

# Model information```

curl http://localhost:8000/model/info

## 📊 Model Performance

# Classes list

curl http://localhost:8000/classes**Training Results:**

```- Dataset: CIFAR-10 (50,000 training, 10,000 test images)
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

## 🔧 Technical Details

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

### API Features

- **Automatic Image Processing**: Resizes any input image to 64×64
- **Multiple Format Support**: JPEG, PNG, BMP, TIFF
- **Batch Processing**: Handle multiple images in a single request
- **Error Handling**: Comprehensive validation and error reporting
- **CORS Support**: Ready for web integration
- **Health Monitoring**: Built-in health checks and logging

### Production Ready

- **Docker Containerization**: Complete containerization with health checks
- **Scalable Architecture**: FastAPI with async support
- **Monitoring**: Built-in logging and health endpoints
- **Security**: Input validation and error handling

## 🧪 Testing

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

## 📋 Assignment Rubric Compliance

| Criteria | Status | Implementation |
|----------|--------|----------------|
| **Code committed to GitHub** | ✅ | Repository with complete implementation |
| **Docker deployment with FastAPI** | ✅ | Dockerfile.assignment2 + API server |
| **API can be queried successfully** | ✅ | Multiple endpoints with comprehensive testing |
| **Well organized code with correct functionality** | ✅ | Modular structure, proper error handling |
| **Conceptual questions answered** | ✅ | Architecture verification and documentation |

## 🎓 Learning Outcomes

This assignment demonstrates:

1. **CNN Architecture Design**: Understanding of convolutional layers, pooling, and fully connected layers
2. **PyTorch Implementation**: Practical deep learning model implementation
3. **API Development**: RESTful API design with FastAPI
4. **Production Deployment**: Docker containerization and deployment strategies
5. **Image Processing**: Handling different image formats and preprocessing
6. **Testing & Validation**: Comprehensive testing strategies

## 🚀 Next Steps

- **Model Improvements**: Experiment with data augmentation, learning rate scheduling
- **Advanced Architectures**: Try ResNet, DenseNet, or custom architectures
- **Production Scaling**: Add load balancing, model versioning, A/B testing
- **Monitoring**: Add performance metrics, model drift detection

## 📞 Support

For questions or issues:
1. Check the API documentation at `http://localhost:8000/docs`
2. Run the demo script: `python assignment2_demo.py`
3. Check the test client: `python test_assignment2_api.py`

---

**Assignment 2: CNN Image Classification API** - Complete implementation ready for submission! 🎉