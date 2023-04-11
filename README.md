# QueryVision
This code implements a book cover image retrieval system using the VGG16 deep learning model. Given a query image, the system retrieves the top 10 book cover images from a dataset of book cover images that are most similar to the query image based on the cosine similarity between their respective feature vectors extracted from the VGG16 model.

## Requirements
Python 3.x
Keras
Scikit-learn
Matplotlib

## Implementation Details
The code first extracts all the book cover images from the zip file and saves them in a directory named "covers". It then loads the pre-trained VGG16 model and uses it to extract the features from each image in the dataset. The features are stored along with the filename of the corresponding image in a list.

Next, the code extracts the features from the query image and calculates the cosine similarity between the query image and each image in the dataset. The similarity scores are sorted in descending order and the top 10 matching images are selected.

Finally, the code displays the top 10 matching images using Matplotlib.
