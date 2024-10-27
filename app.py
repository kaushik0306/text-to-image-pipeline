from flask import Flask, request, jsonify, send_from_directory
import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import logging
from diffusers import StableDiffusionPipeline
from segment_anything import SamPredictor, sam_model_registry
from flask_cors import CORS
import os
import requests
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from any origin

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set device to CPU
device = torch.device("cpu")

# Load Stable Diffusion model for the /generate endpoint
logging.info("Loading Stable Diffusion model...")
model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
logging.info("Stable Diffusion model loaded successfully!")

# Load CLIP model for the /analyze endpoint
logging.info("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
logging.info("CLIP model loaded successfully!")

# Initialize SAM2 for image segmentation
logging.info("Loading SAM2 model...")
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(device)
sam_predictor = SamPredictor(sam)
logging.info("SAM2 model loaded successfully!")

# Ensure the directory for generated images exists
if not os.path.exists('/app/generated_images'):
    os.makedirs('/app/generated_images')

# Endpoint to generate image using Stable Diffusion
@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        data = request.get_json()  # Ensure proper JSON handling
        if data is None:
            logging.error("No JSON data received")
            return jsonify({"error": "No JSON data received"}), 400
        
        prompt = data.get("prompt", "")
        
        # Validate the prompt
        if not prompt or not isinstance(prompt, str):
            logging.error("Invalid or missing prompt")
            return jsonify({"error": "Prompt must be a valid non-empty string"}), 400
        
        logging.info(f"Generating image for prompt: {prompt}")
        
        # Generate the image
        with torch.no_grad():
            image = model(prompt).images[0]

        # Save the image
        image_path = "/app/generated_images/generated_image.png"
        image.save(image_path)

        # Generate the public URL for the image
        public_ip = "54.89.36.139"  # Replace this with your EC2 instance public IP
        image_url = f"http://{public_ip}:5000/images/generated_image.png"
        
        logging.info(f"Image generated and can be accessed at: {image_url}")
        return jsonify({"message": "Image generated successfully!", "image_url": image_url})

    except Exception as e:
        logging.error(f"Error in /generate: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Serve static files for the generated images
@app.route('/images/<filename>')
def get_image(filename):
    try:
        return send_from_directory('/app/generated_images', filename)
    except Exception as e:
        logging.error(f"Error serving image {filename}: {str(e)}")
        return jsonify({"error": "Image not found"}), 404


# Endpoint to analyze the image using CLIP and SAM2
@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        logging.info("Received request to /analyze")
        data = request.get_json()
        image_url = data.get("image_url", "")

        # Download the image from the URL
        if not image_url:
            logging.error("No image URL provided")
            return jsonify({"error": "No image URL provided"}), 400

        try:
            logging.info(f"Attempting to download image from: {image_url}")
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            logging.error(f"Failed to download or open image: {str(e)}")
            return jsonify({"error": "Failed to download or open image"}), 400

        # Convert PIL image to NumPy array, then to a tensor for SAM2
        image_array = np.array(image)
        image_tensor = torch.tensor(image_array).permute(2, 0, 1).float()  # Convert to tensor and adjust dimensions

        # Run SAM2 segmentation
        try:
            logging.info("Running image segmentation with SAM2")
            sam_predictor.set_image(image_tensor)
            masks, scores, _ = sam_predictor.predict()  # Perform segmentation
            logging.info("Segmentation completed")
        except Exception as e:
            logging.error(f"Error during SAM2 segmentation: {str(e)}")
            return jsonify({"error": "Error during segmentation"}), 500

        # Run CLIP analysis on the entire image
        logging.info("Running CLIP analysis on the entire image")
        text_prompts = ["a sunset", "mountains", "a river", "a forest", "a city"]
        inputs = clip_processor(text=text_prompts, images=image, return_tensors="pt", padding=True).to(device)
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        prob_values = probs.cpu().detach().numpy().tolist()[0]  # Flatten the output

        # Prepare the response in the desired format
        response = {
            "request_id": "unique_id",  # You can generate a unique ID here
            "generated_image": None,  # Set to None or a placeholder if not applicable
            "clip_analysis": {
                "concepts": text_prompts,
                "confidence_scores": prob_values
            },
            "basic_segmentation": {
                "masks": masks.tolist(),  # Include the masks obtained from SAM2
                "scores": scores.tolist()  # Segmentation confidence scores
            }
        }

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error in /analyze: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
