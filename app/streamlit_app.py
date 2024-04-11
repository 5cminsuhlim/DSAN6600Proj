import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import logging

#setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#this function loads the trained model
def load_model(model_path):
    logging.info(f"Loading model from {model_path}")
    model = models.resnet18(pretrained=False) #assumes resnet18
    model.load_state_dict(torch.load(model_path))
    model.eval()
    logging.info("Model loaded and set to evaluation mode")
    return model

#preprocess the image
def preprocess_image(image):
    logging.info("Preprocessing the uploaded image")
    transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    processed_image = transform(image).unsqueeze(0)
    logging.info("Image preprocessed successfully")
    return processed_image

#this function selects an image from the vector store based on the model's output
def select_image_from_vector_store(prediction):
    # Implement your logic to select and return an image path from the vector store
    # For simplicity, this function just returns a fixed path
    logging.info("Selecting an image from the vector store based on the model's prediction")
    return 'code/images/sample_image.jpg'

def main():
    st.title("Your Deep Learning App")

    #load the model
    model = load_model('models/resnet1.pth')

    #UI for image upload
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        logging.info("Image uploaded by user")
        image = Image.open(uploaded_file)

        #display the uploaded image
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        logging.info("Uploaded image displayed")

        #preprocess the image
        processed_image = preprocess_image(image)

        #run inference
        logging.info("Running inference on the preprocessed image")
        with torch.no_grad():
            prediction = model(processed_image)
        logging.info("Inference completed")

        # For simplicity, the prediction result is not used to select the image. You can implement your own logic here.
        selected_image_path = select_image_from_vector_store(prediction)

        #display the selected image from the vector store
        vector_store_image = Image.open(selected_image_path)
        st.image(vector_store_image, caption='Selected Image from Vector Store.', use_column_width=True)
        logging.info("Displayed selected image from vector store")

if __name__ == "__main__":
    main()
