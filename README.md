# QueryVision
This code implements a book cover image retrieval system using the VGG16 deep learning model. Given a query image, the system retrieves the top 10 book cover images from a dataset of book cover images that are most similar to the query image based on the cosine similarity between their respective feature vectors extracted from the VGG16 model.

## Background
This is an update on the project that I did during my undergraduate final semester project. Then, I employed MATLAB to develop an image retrieval system that utilized Speeded Up Robust Features (SURF) for feature extraction. The system was designed to compare the extracted features from an input query image with those of a database of images and return the top 10 images with the highest similarity scores. This project aimed to enhance image retrieval efficiency and accuracy by utilizing advanced techniques in image processing and computer vision. By leveraging SURF features, the system was able to provide more accurate and reliable image retrieval results, making it a valuable tool for various applications such as medical imaging, security surveillance, and e-commerce.

## Requirements
- Python 3.x
- Keras
- Scikit-learn
- Matplotlib

## Implementation Details
The code first extracts all the book cover images from the zip file and saves them in a directory named "covers". It then loads the pre-trained VGG16 model and uses it to extract the features from each image in the dataset. The features are stored along with the filename of the corresponding image in a list.

Next, the code extracts the features from the query image and calculates the cosine similarity between the query image and each image in the dataset. The similarity scores are sorted in descending order and the top 10 matching images are selected.

Finally, the code displays the top 10 matching images using Matplotlib.

## Future work
I believe that there are several ways to improve this project. 

- Using a larger dataset would be beneficial as the current dataset is relatively small, leading to overfitting and limited generalization. By using a diverse set of book covers in a larger dataset, the model's performance can be improved.

- Fine-tuning the VGG16 model would be helpful. Although it is pre-trained on the ImageNet dataset, fine-tuning it on a specific book cover dataset can increase its accuracy on this task.

- Experimenting with other pre-trained models like ResNet, Inception, or EfficientNet can also help to improve the model's performance.

- Augmenting the dataset using techniques like rotation, cropping, and flipping can increase the dataset's size and improve the model's performance.

- Instead of using cosine similarity as the similarity metric between images, other similarity metrics like Euclidean distance, Manhattan distance, or Pearson correlation coefficient can be experimented with to see if they perform better.

- Using transfer learning instead of pre-trained models can lead to better accuracy and performance.
