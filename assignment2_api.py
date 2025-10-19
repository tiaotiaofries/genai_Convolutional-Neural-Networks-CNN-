"""
Assignment 2: FastAPI Image Classification Server
API endpoints for the trained CNN model

This FastAPI server provides:
1. Image classification endpoint
2. Model information endpoint  
3. Health check endpoint
4. Batch prediction endpoint
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
from typing import List, Optional
import logging
from datetime import datetime

# Import our model
from assignment2_cnn_classifier import AssignmentCNN
from helper_lib import get_device, load_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Assignment 2 CNN Image Classifier",
    description="Image classification API using the Assignment 2 CNN architecture",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
device = None
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Image preprocessing transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to 64x64 as required
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

@app.on_event("startup")
async def startup_event():
    """Load the trained model on startup"""
    global model, device
    
    try:
        device = get_device()
        logger.info(f"Using device: {device}")
        
        # Load the trained model
        model = AssignmentCNN(num_classes=10).to(device)
        model = load_model(model, 'models/assignment2_cnn.pth')
        model.eval()
        
        logger.info("Model loaded successfully!")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Assignment 2 CNN Image Classifier API",
        "version": "1.0.0",
        "model": "Assignment 2 CNN",
        "input_size": "64x64 RGB images",
        "classes": len(class_names),
        "endpoints": {
            "/classify": "Upload image for classification",
            "/classify/batch": "Upload multiple images for batch classification",
            "/model/info": "Get model architecture information",
            "/health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.get("/model/info")
async def model_info():
    """Get model architecture and information"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Get architecture summary
    arch_summary = model.get_architecture_summary()
    
    return {
        "model_name": "Assignment 2 CNN",
        "architecture": arch_summary,
        "classes": class_names,
        "input_requirements": {
            "format": "RGB image",
            "size": "Any size (will be resized to 64x64)",
            "channels": 3,
            "supported_formats": ["JPEG", "PNG", "BMP", "TIFF"]
        },
        "model_file": "models/assignment2_cnn.pth"
    }

def preprocess_image(image_data: bytes) -> torch.Tensor:
    """Preprocess uploaded image for model inference"""
    try:
        # Open image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Classify a single uploaded image
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON response with prediction results
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess image
        image_data = await file.read()
        image_tensor = preprocess_image(image_data)
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Get top 3 predictions
            top3_prob, top3_indices = torch.topk(probabilities, 3, dim=1)
            
        # Format response
        response = {
            "filename": file.filename,
            "prediction": {
                "class": class_names[predicted.item()],
                "class_index": predicted.item(),
                "confidence": float(confidence.item())
            },
            "top_3_predictions": [
                {
                    "class": class_names[idx.item()],
                    "class_index": idx.item(),
                    "confidence": float(prob.item())
                }
                for prob, idx in zip(top3_prob[0], top3_indices[0])
            ],
            "model_info": {
                "architecture": "Assignment 2 CNN",
                "input_size": "64x64 RGB",
                "total_classes": len(class_names)
            }
        }
        
        logger.info(f"Classified {file.filename}: {class_names[predicted.item()]} ({confidence.item():.3f})")
        
        return response
        
    except Exception as e:
        logger.error(f"Error during classification: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/classify/batch")
async def classify_batch(files: List[UploadFile] = File(...)):
    """
    Classify multiple uploaded images in batch
    
    Args:
        files: List of uploaded image files
        
    Returns:
        JSON response with batch prediction results
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    results = []
    
    for file in files:
        # Validate file type
        if not file.content_type.startswith('image/'):
            results.append({
                "filename": file.filename,
                "error": "File must be an image"
            })
            continue
        
        try:
            # Read and preprocess image
            image_data = await file.read()
            image_tensor = preprocess_image(image_data)
            image_tensor = image_tensor.to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Add result
            results.append({
                "filename": file.filename,
                "prediction": {
                    "class": class_names[predicted.item()],
                    "class_index": predicted.item(),
                    "confidence": float(confidence.item())
                }
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "batch_size": len(files),
        "results": results,
        "model_info": {
            "architecture": "Assignment 2 CNN",
            "input_size": "64x64 RGB"
        }
    }

@app.get("/classes")
async def get_classes():
    """Get list of supported classes"""
    return {
        "classes": class_names,
        "total_classes": len(class_names),
        "class_mapping": {i: name for i, name in enumerate(class_names)}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)