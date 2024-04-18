import streamlit as st
from PIL import Image
import app_functions as af
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    # Set page config to wide
    st.set_page_config(layout="wide")

    # Set title and description
    st.title("WikiArt Image Similarity Finder 🖼️")
    st.write(
        "Upload an image to find similar images. Please use images of people; horizontal images work best!"
    )

    col1, col2 = st.columns(spec=[0.25, 0.75])

    # Set up vector store to pretty name dict
    vs_names = {
        "threshold_vs": "Thresholded Subject Mask",
        "segmented_edge_vs": "Subject Segmented Edge Detection",
        "dimmed_vs": "Dimmed Original Image",
        "edge_vs": "Edge Detection",
        "mask_vs": "Subject Mask",
        "raw_vs": "Original Image",
    }

    # Set uploading file to False
    real_file = False

    # Image upload widget
    with col1.container(height=600):
        uploaded_file = st.file_uploader(
            "Choose an image:", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

            # User input for number of similar images
            k = st.number_input(
                "Enter the number of similar images you want to find:",
                min_value=1,
                value=1,
                step=1,
            )

            # Preprocessing type selection
            preprocessing_type = st.multiselect(
                label="Select the preprocessing type(s):",
                options=list(vs_names.values()),
                default=list(vs_names.values()),  # Select all by default
            )

            # If none selected, select all
            if preprocessing_type == []:
                preprocessing_type = list(vs_names.values())
                st.write("No preprocessing type selected. Selecting all.")

            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            logging.info("Image displayed successfully.")

            # Set real_file to True
            real_file = True

    with col2.container(height=600):
        if st.button("Find Similar Images") and real_file:
            try:

                st.write("Processing image...")

                # Save uploaded image to a user_inputs folder
                input_folder = "data/user_inputs/"
                os.makedirs(input_folder, exist_ok=True)
                input_path = os.path.join(input_folder, uploaded_file.name)
                image.save(input_path)
                logging.info(f"Image saved to: {input_path}")

                image_widths = int(750 / k)
                logging.info(f"Image width set to {image_widths}.")

                # Process image to get similar images
                results = af.get_all_results(input_path, k)
                logging.info("Similar images found.")

                # Display similar images
                if results is not None and "results" in results:

                    for key in results["results"]:

                        # If the vector store is not selected, skip
                        if vs_names[key] not in preprocessing_type:
                            continue

                        st.subheader(f"Similar Images from {vs_names[key]}")
                        similar_images = results["results"][key]["images"]
                        similarities = results["results"][key]["similarities"]

                        # Temporary directory to store images
                        temp_dir = "data/temp_images"
                        os.makedirs(temp_dir, exist_ok=True)

                        image_paths = []

                        # Write images to temporary directory and display them
                        for idx, (sim_image, similarity) in enumerate(
                            zip(similar_images, similarities)
                        ):
                            # Write the image to a temporary file
                            temp_image_path = os.path.join(
                                temp_dir, f"temp_image_{idx}.png"
                            )
                            Image.fromarray(sim_image).save(temp_image_path)

                            # add the image path to the list
                            image_paths.append(temp_image_path)

                        # Display the images
                        st.image(
                            image_paths,
                            caption=[
                                f"Similarity {similarity:.3f}"
                                for similarity in similarities
                            ],
                            width=image_widths,
                        )

                        # Delete the temporary directory and its contents
                        for filename in os.listdir(temp_dir):
                            file_path = os.path.join(temp_dir, filename)
                            os.remove(file_path)
                        os.rmdir(temp_dir)

                # Delete the uploaded image when the app is closed
                os.remove(input_path)
                logging.info(f"Deleted uploaded image: {input_path}")

            except Exception as e:
                st.error(f"Error processing image: {e}")
                logging.error(f"Error processing image: {e}")


# Entry point for Streamlit app
if __name__ == "__main__":
    main()
