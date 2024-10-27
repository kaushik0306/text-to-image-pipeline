
# Text-to-Image Generation with Multi-Model Analysis

This project implements a text-to-image generation pipeline using Stable Diffusion, CLIP analysis, and SAM2 segmentation. The application generates images from text prompts and performs analysis on the generated images.

## Features
- **Text-to-Image Generation**: Uses Stable Diffusion to generate images based on text prompts.
- **Image Analysis**: Uses CLIP to perform concept analysis on the generated image.
- **Image Segmentation**: Uses SAM2 for basic segmentation analysis.
- **API Endpoints**: Provides `/generate` and `/analyze` endpoints for image generation and analysis.

## Getting Started

### Prerequisites
- Docker
- Python 3.10+
- pip (Python package installer)

### Setup
1. **Clone the Repository**

   git clone https://github.com/kaushik0306/text-to-image-pipeline.git
   cd text-to-image-pipeline
  

2. **Install Dependencies**
  
   pip install -r requirements.txt
   

3. **Set Up Docker**
   - Build and run the Docker container:
     
     docker build -t text-to-image-app .
     docker run -p 5000:5000 text-to-image-app
     

4. **Environment Setup**
   - Make sure you have a virtual environment:
     
     python3 -m venv myenv
     source myenv/bin/activate  # For Linux/Mac
     myenv\Scripts\activate  # For Windows
     

5. **Running the Application**
   - After setting up the environment, you can run the app locally:
    
     python app.py
    

## API Endpoints

### 1. `/generate`
- **Method**: `POST`
- **Description**: Generates an image based on the provided text prompt.
- **Request Body**: 
  ```json
  {
    "prompt": "A beautiful sunset over the mountains"
  }
  ```
- **Response**:
  ```json
  {
    "message": "Image generated successfully!",
    "image_url": "http://<your-ip>:5000/images/generated_image.png"
  }
  ```

### 2. `/analyze`
- **Method**: `POST`
- **Description**: Analyzes the generated image using CLIP and SAM2 segmentation.
- **Request Body**: 
  ```json
  {
    "image_url": "http://<your-ip>:5000/images/generated_image.png"
  }
  ```
- **Response**:
  ```json
  {
    "request_id": "unique_id",
    "clip_analysis": {
        "concepts": ["a sunset", "mountains", "a river"],
        "confidence_scores": [0.9, 0.85, 0.75]
    },
    "basic_segmentation": {
        "masks": [[...]],
        "scores": [0.8, 0.5, 0.3]
    }
  }
  ```

## Testing
- **Running Tests**
   ```bash
   pytest tests/
   ```
- The tests cover:
   - Image generation with valid/invalid prompts.
   - Image analysis with valid/invalid image paths.
   


---

Including a README file provides a friendly introduction and quick access to the most important project details. While the full documentation can go into depth, the README should act as an easy-to-find, quick-start guide for new users.
