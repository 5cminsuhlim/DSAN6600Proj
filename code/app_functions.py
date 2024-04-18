"""
import packages:
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import deeplake
from torchvision import transforms, models, datasets
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
import torchvision
import flask
from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore
import os
import torch
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image, ImageFilter
import shutil
import multiprocess as mp
from tqdm import tqdm
import warnings


# Functions to Process Images
def edge_detect(image_path):

    # load image from path
    image = Image.open(image_path)
    # convert to greyscale as edge detection needs greyscale
    image = image.convert("L")
    # detect edges
    image = image.filter(ImageFilter.FIND_EDGES)

    return image, image_path  ## Why are we returning the path?


def person_highlighter(image_path, threshold=0.5, dimming_factor=0.3, all_images=False):
    img = Image.open(image_path)
    image_copy = img.copy()
    x = torchvision.transforms.functional.to_tensor(img)

    # Load weights
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights, pretrained=True)

    # Set the model to evaluation mode
    preprocess = weights.transforms()

    batch = preprocess(x).unsqueeze(0)

    # Perform inference
    prediction = model(batch)["out"]
    normalized_masks = prediction.softmax(dim=1)

    class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
    original_mask = normalized_masks[0, class_to_idx["person"]]

    # Add zeroes to the mask so it's the same size as the image
    mask = (
        torch.nn.functional.interpolate(
            original_mask.unsqueeze(0).unsqueeze(0),
            size=(img.height, img.width),
            mode="nearest",
        )
        .squeeze(0)
        .squeeze(0)
    )

    # Convert the mask to a numpy array
    mask = mask.detach().numpy()

    person_mask = mask > threshold

    dimmed_img = image_copy.copy()

    # Make dimmed img the same size as the person mask
    dimmed_img = np.array(img)

    # Multiply the dimmed image by the mask

    # This is much faster but struggles with certain sized photos
    try:
        dimmed_img[~person_mask] = dimmed_img[~person_mask] * dimming_factor

    # This is slower but works for all photos
    except IndexError:

        for y in range(img.size[1]):
            for x in range(img.size[0]):
                if not person_mask[y, x]:
                    dimmed_img[y, x] = dimmed_img[y, x] * dimming_factor

    # If all images, Return threshold mask, mask, and dimmed image
    if all_images:
        return (
            to_pil_image(original_mask),
            Image.fromarray(person_mask),
            Image.fromarray(dimmed_img),
        )

    # Return only the dimmed image
    else:
        return Image.fromarray(dimmed_img)


def resnet():
    # Check if GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the model
    model = models.resnet34(pretrained=True)

    # Create a feature extractor
    return_nodes = {"avgpool": "embedding"}
    model = create_feature_extractor(model, return_nodes=return_nodes)

    # Freeze the model
    model.eval()
    model.to(device)
    return model


model = resnet()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def embedding_function(images, model=model, device=device, batch_size=4):
    """Creates a list of embeddings based on a list of image filenames. Images are processed in batches."""
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: torch.cat([x, x, x], dim=0) if x.shape[0] == 1 else x
            ),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if isinstance(images, str):
        images = [images]

    # Proceess the embeddings in batches, but return everything as a single list
    embeddings = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack(
            [transform(Image.open(item)) for item in images[i : i + batch_size]]
        )
        batch = batch.to(device)
        with torch.no_grad():
            embeddings += model(batch)["embedding"][:, :, 0, 0].cpu().numpy().tolist()

    return embeddings


# Functions to get results
## Based on get_real_results in consolidate.ipynb
# Now it will take the image path and the vector store and return the resulting images as a json
def get_images(image_path, vector_store, ds, k=5):
    # Get results from vector store
    result = vector_store.search(
        embedding_data=image_path, embedding_function=embedding_function, k=k
    )

    # Get image name of the first image
    image_name = os.path.basename(image_path)

    # Get the original image path
    image_name = image_name.split("/")[-1]

    # If the image is a test image, get the path from the test-images folder
    if "test-images" in image_path:
        image_path = f"../data/test-images/{image_name}"  # Change to be wherever we are storing user images if needed

    # Else it will be under the same base name in test-images
    else:
        image_name = image_name.split("_")[-1]
        image_path = f"../data/test-images/{image_name}"

    # Create a list to store the results
    images = []
    similarities = []
    labels = []

    # Get the top k results
    for i in range(k):
        # Set crop to false
        crop = False

        # Get image data
        label = str(result["filename"][i])

        # Get image index
        name = label.split("/")[-1]

        # Remove the .jpg
        name = name.split(".")[0]
        label = label.split("/")[-1]
        label = label.split(".")[0]

        # Split after index
        name = name.split("index_")[-1]

        # Check if bounding_box or subject_box is in the name
        if "bounding_box" in name:
            name = name.split("_bounding_box")[0]

            # Get bounding box info
            coords = label.split("_bounding_box_")[-1].split("_")

            # Get the coordinates
            x1 = int(coords[0])
            y1 = int(coords[1])
            x2 = int(coords[2])
            y2 = int(coords[3])

            # Set crop to true
            crop = True

        elif "subject_box" in name:
            name = name.split("_subject_box")[0]

            # Get bounding box info
            coords = label.split("_subject_box_")[-1].split("_")

            # Get the coordinates
            x1 = int(coords[0])
            y1 = int(coords[1])
            x2 = int(coords[2])
            y2 = int(coords[3])

            # Set crop to true
            crop = True

        # Change name to int
        name = int(name)

        # Get image from ds
        img = ds.images[name].numpy()

        # If crop is true, crop the image
        if crop:
            img = img[y1:y2, x1:x2]

        # Get similarity
        similarity = result["score"][i]

        # Get label
        label = ds.labels[name].data()["text"][0]

        # Add to lists
        images.append(img)
        similarities.append(similarity)
        labels.append(label)

    # Create output dictionary
    output = {"images": images, "similarities": similarities, "labels": labels}

    return output


# This is the big function, for streamlit, you need to save the user image to a path and then pass it to this function
# Ideally, save it to the test-images folder so that the functions are already set up to handle it
# NOTE: Do not include underscores in the image path as it will mess with the code
def get_all_results(image_path, k=5):

    # Load dataset
    ds = deeplake.load("hub://activeloop/wiki-art", read_only=True, verbose=False)

    # Hide warnings with: UserWarning: The default value of the antialias parameter of all the resizing
    warnings.filterwarnings(
        "ignore",
        message="The default value of the antialias parameter of all the resizing *",
    )

    # Get all vector stores
    vector_stores = os.listdir("../data/vector_stores")

    # Remove the .DS_Store file
    if ".DS_Store" in vector_stores:
        vector_stores.remove(".DS_Store")

    # Create subject segmentation
    full_mask, threshold_mask, dimmed_img = person_highlighter(
        image_path,
        threshold=0.1,
        dimming_factor=0.3,
        all_images=True,  # We need to establish a threshold and dimming factor
    )

    # Create edge detection
    edge_img, edge_img_path = edge_detect(
        image_path
    )  # Remove path from return if we want

    # Create segmented edge detection
    # Make dimmed img the same size as the person mask
    dimmed_edge = np.array(edge_img)
    threshold_array = np.array(threshold_mask)

    # Multiply the dimmed image by the mask

    # This is much faster but struggles with certain sized photos
    try:
        dimmed_edge[~threshold_array] = dimmed_edge[~threshold_array] * 0.3

    # This is slower but works for all photos
    except IndexError:

        for y in range(dimmed_edge.size[1]):
            for x in range(dimmed_edge.size[0]):
                if not dimmed_edge[y, x]:
                    dimmed_edge[y, x] = dimmed_edge[y, x] * 0.3

    # Convert the dimmed edge to an image
    dimmed_edge = Image.fromarray(dimmed_edge)

    # Create tmp folder
    os.makedirs("tmp", exist_ok=True)

    # Save the images
    image_name = os.path.basename(image_path)
    edge_img.save(f"tmp/edge_{image_name}")
    dimmed_img.save(f"tmp/dimmed_{image_name}")
    threshold_mask.save(f"tmp/threshold_{image_name}")
    full_mask.save(f"tmp/mask_{image_name}")
    dimmed_edge.save(f"tmp/segmented_edge_{image_name}")

    # Load original image
    original_image = Image.open(image_path)

    # Create output dictionary
    output = {
        "inputs": {
            "original_image": original_image,
            "edges": edge_img,
            "dimmed": dimmed_img,
            "threshold": threshold_mask,
            "mask": full_mask,
            "segmented_edge": dimmed_edge,
        },
        "results": {},
    }

    # Get results for each vector store
    for vs in vector_stores:
        vector_store_path = f"../data/vector_stores/{vs}"
        vector_store = VectorStore(
            path=vector_store_path, verbose=False, read_only=True
        )  # Can't figure out how to hide the message from this

        # Make sure to use the correct image for the correct vector store
        ### For the final version, we can change it to assign these results to a variable and then return them to display
        try:
            if vs == "raw_vs":
                raw_results = get_images(image_path, vector_store, ds, k)
                output["results"]["raw_vs"] = raw_results
            elif vs == "edge_vs":
                edge_results = get_images(f"tmp/edge_{image_name}", vector_store, ds, k)
                output["results"]["edge_vs"] = edge_results
            elif vs == "dimmed_vs":
                dimmed_results = get_images(
                    f"tmp/dimmed_{image_name}", vector_store, ds, k
                )
                output["results"]["dimmed_vs"] = dimmed_results
            elif vs == "threshold_vs":
                threshold_results = get_images(
                    f"tmp/threshold_{image_name}", vector_store, ds, k
                )
                output["results"]["threshold_vs"] = threshold_results
            elif vs == "mask_vs":
                mask_results = get_images(f"tmp/mask_{image_name}", vector_store, ds, k)
                output["results"]["mask_vs"] = mask_results
            elif vs == "segmented_edge_vs":
                segmented_edge_results = get_images(
                    f"tmp/segmented_edge_{image_name}", vector_store, ds, k
                )
                output["results"]["segmented_edge_vs"] = segmented_edge_results
        except Exception as e:
            print(f"Error with {vs}")
            print(e)

    # Remove tmp folder
    shutil.rmtree("tmp")

    return output
