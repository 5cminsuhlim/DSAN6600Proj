import streamlit as st
from PIL import Image
import app_functions as af
import logging
import os
import matplotlib.pyplot as plt
import numpy as np

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

    # Create columns at 1:3 ratio
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

    # Reverse dict for pretty name to vector store
    vs_names_rev = {v: k for k, v in vs_names.items()}

    # Create Short names to add to images
    short_names = {
        "Thresholded Subject Mask": "Theresholded",
        "Subject Segmented Edge Detection": "Segmented Edge",
        "Dimmed Original Image": "Dimmed",
        "Edge Detection": "Edge Detection",
        "Subject Mask": "Subject Mask",
        "Original Image": "Raw Image",
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

            # Save preprocessing type as a global variable
            st.session_state.preprocessing_type = preprocessing_type

            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            logging.info("Image displayed successfully.")

            # Set real_file to True
            real_file = True

    # Bottom half of column 1 to display the preprocessed images
    with col1.container(height=600):
        if st.button("View Preprocessed Images"):
            # Get results inputs
            results = st.session_state.results
            inputs = results["inputs"]

            # Input keys are ['original_image', 'edges', 'dimmed', 'threshold', 'mask', 'segmented_edge']
            # Create dict to map keys to pretty names
            input_names = {
                "original_image": "Original Image",
                "edges": "Edge Detection",
                "dimmed": "Dimmed Original Image",
                "threshold": "Thresholded Subject Mask",
                "mask": "Subject Mask",
                "segmented_edge": "Subject Segmented Edge Detection",
            }
            input_names_rev = {v: k for k, v in input_names.items()}

            # Display the preprocessed images
            for processing in preprocessing_type:
                # Skip the original image since we already displayed it
                if processing == "Original Image":
                    continue

                logging.info(f"Preprocessing image for {processing}...")

                # Get keys for the preprocessing type
                key = input_names_rev[processing]

                # Get the preprocessed image
                preprocessed_image = inputs[key]

                # Display the preprocessed image
                st.image(preprocessed_image, caption=processing, use_column_width=True)

    # Column 2 to display the similar images
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

                # Get the image width based on the number of similar images
                image_widths = int(750 / k)
                logging.info(f"Image width set to {image_widths}.")

                # Process image to get similar images
                results = af.get_all_results(input_path, k)

                # Save results as a global variable
                st.session_state.results = results

                logging.info("Similar images found.")

                # Display similar images
                if results is not None and "results" in results:

                    for key in results["results"]:

                        # If the vector store is not selected, skip
                        if vs_names[key] not in preprocessing_type:
                            continue

                        # Get images and similarities
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

            except Exception as e:
                logging.error(f"Error processing image: {e}")
                st.write(
                    f"An error occurred in creating your image. Please refresh the page and try again; the results will be the same as long as you upload the same image."
                )

    # Create output options in the bottom half of column 2
    with col2.container(height=600):
        # Preprocessing type selection
        output_type = st.multiselect(
            label="Select the output(s) you want:",
            options=list(vs_names.values()),
            default=list(vs_names.values()),  # Select all by default
        )

        # Create temporary directory to store images
        temp_dir = "data/temp_images"
        os.makedirs(temp_dir, exist_ok=True)

        # Create output image with matplotlib
        if (
            st.button("Create Output Image")
            and real_file
            and st.session_state.results is not None
        ):
            st.write("Creating output image...")
            try:
                # Get the results
                results = st.session_state.results

                rows = len(output_type)
                row = 0
                cols = k

                # Get the output images for multiple output types
                if rows > 1:

                    # Create a figure to display the output images
                    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))

                    for processing in output_type:
                        logging.info(f"Creating output image for {processing}...")

                        # Get key
                        key = vs_names_rev[processing]

                        # Get images and similarities
                        output_images = results["results"][key]["images"]
                        similarities = results["results"][key]["similarities"]

                        for i in range(cols):
                            image_arr = np.array(output_images[i])
                            axs[row, i].imshow(image_arr)
                            axs[row, i].axis("off")
                            axs[row, i].set_title(
                                f"{short_names[processing]} {i + 1}\nsimilarity: {similarities[i]:.3f}"
                            )

                        row += 1

                elif rows == 1:
                    fig, axs = plt.subplots(1, cols, figsize=(10, 10))

                    # Get key
                    processing = output_type[0]
                    key = vs_names_rev[processing]

                    # Get images
                    output_images = results["results"][key]["images"]
                    similarities = results["results"][key]["similarities"]

                    for i in range(cols):
                        image_arr = np.array(output_images[i])
                        axs[i].imshow(image_arr)
                        axs[i].axis("off")
                        axs[i].set_title(
                            f"{short_names[processing]} {i + 1}\nsimilarity: {similarities[i]:.3f}"
                        )

                    row += 1

                else:
                    st.write("No output type selected - assuming to output all")

                    output_type = vs_names.values()

                    rows = len(output_type)
                    row = 0
                    cols = k

                    # Create a figure to display the output images
                    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))

                    for processing in output_type:
                        logging.info(f"Creating output image for {processing}...")

                        # Get key
                        key = vs_names_rev[processing]

                        # Get images and similarities
                        output_images = results["results"][key]["images"]
                        similarities = results["results"][key]["similarities"]

                        for i in range(cols):
                            image_arr = np.array(output_images[i])
                            axs[row, i].imshow(image_arr)
                            axs[row, i].axis("off")
                            axs[row, i].set_title(
                                f"{short_names[processing]} {i + 1}\nsimilarity: {similarities[i]:.3f}"
                            )

                        row += 1

                # Tight layout
                plt.tight_layout()

                # Save the figure
                output_path = temp_dir + "/output_image.png"
                plt.savefig(output_path)

                # Create 2, 1 grid
                fig, ax = plt.subplots(1, 2, figsize=(10, 10))

                # Original image on the left
                ax[0].imshow(image)
                ax[0].axis("off")
                ax[0].set_title("Uploaded Image")

                # Output image on the right
                output_image = Image.open(output_path)
                ax[1].imshow(output_image)
                ax[1].axis("off")
                ax[1].set_title("Outputs")

                # Display the figure
                st.pyplot(fig)

                # Save the final image
                fig.savefig(temp_dir + "/final_image.png")

                # Create download link for the output image
                with open("data/temp_images/final_image.png", "rb") as final_image:
                    st.download_button(
                        label="Download Output Image",
                        data=final_image,
                        file_name="output_image.png",
                        mime="image/png",
                    )

                # Delete the output image
                os.remove(output_path)
                logging.info(f"Deleted output image: {output_path}")

            except Exception as e:
                st.error(f"Error creating output image: {e}")
                logging.error(f"Error creating output image: {e}")

            # Delete the temporary directory and its contents
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                os.remove(file_path)
            os.rmdir(temp_dir)

            # Delete the user_inputs contents
            for filename in os.listdir("data/user_inputs"):
                file_path = os.path.join("data/user_inputs", filename)
                os.remove(file_path)


# Entry point for Streamlit app
if __name__ == "__main__":
    main()
