import streamlit as st
from PIL import Image
import app_functions
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    st.title("Wikiart Image Similarity Finder")
    st.write("Upload an image to find similar images.")

    # Image upload widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')

        # User input for number of similar images
        k = st.number_input("Enter the number of similar images you want to find:", min_value=1, value=1, step=1)

        if st.button("Find Similar Images"):
            try:
                # Display the uploaded image
                st.image(image, caption='Uploaded Image', use_column_width=True)
                logging.info("Image displayed successfully.")

                st.write("Processing image...")

                # Save uploaded image to a user_inputs folder
                input_folder = 'data/user_inputs/'
                os.makedirs(input_folder, exist_ok=True)
                input_path = os.path.join(input_folder, uploaded_file.name)
                image.save(input_path)
                logging.info(f"Image saved to: {input_path}")

                # Process image to get similar images
                results = app_functions.get_all_results(input_path, k)
                logging.info("Similar images found.")

                # Display similar images
                if results is not None and "results" in results:
                    for key in results["results"]:
                        st.subheader(f"Similar Images from {key}")
                        similar_images = results["results"][key]["images"]
                        similarities = results["results"][key]["similarities"]

                        for idx, (sim_image, similarity) in enumerate(zip(similar_images, similarities)):
                            st.image(sim_image, caption=f"Similarity {similarity:.3f}", use_column_width=True)
                            logging.info(f"Displayed similar image {idx + 1} from {key}")

            except Exception as e:
                st.error(f"Error processing image: {e}")
                logging.error(f"Error processing image: {e}")

# Entry point for Streamlit app
if __name__ == "__main__":
    main()
