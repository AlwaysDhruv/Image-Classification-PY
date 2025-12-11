# Image Classification using CNNs

## About The Project

ObjectTorch is a Python-based image classification tool that allows you to train and test your own custom models. It uses TensorFlow and Keras to build and train a convolutional neural network (CNN) for image classification. The project provides a simple command-line interface to manage your models.

![1_7_BCJFzekmPXmJQVRdDgwg](https://github.com/user-attachments/assets/f9ac7351-7472-4ea7-89ae-ab69868457a7)

## Features

*   **Train Custom Models:** Train your own image classification models on your datasets.
*   **Test Models:** Use your trained models to make predictions on new images.
*   **Text-to-Speech:** Get audible feedback for predictions.
*   **Simple CLI:** Easy-to-use command-line interface for managing models.

## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

*   Python 3.12
*   pip

### Installation

1.  Clone the repo
    ```sh
    git clone https://github.com/your_username/your_repository.git
    ```
2.  Install PIP packages
    ```sh
    pip install -r requirements.txt
    ```

## Usage

Run the `Leading.py` script to start the application:

```sh
python Leading.py
```

You will be presented with the following options:

*   **Test The Model (1):** Load a pre-trained model and classify an image.
*   **Create The Model (2):** Train a new model on a dataset.
*   **Delete The Model (3):** Remove a saved model.
*   **Clear The Screen (*):** Clear the console.
*   **Exit (0):** Close the application.

### Training a New Model

1.  Choose option `2` to create a new model.
2.  Select the directory containing your dataset. The dataset should be organized into subdirectories, where each subdirectory represents a class.
3.  Enter the number of epochs you want to train the model for.
4.  Once training is complete, you can save the model to a specified location.

### Testing a Model

1.  Choose option `1` to test a model.
2.  Select the directory where your saved model (`.keras` file) and class mapping (`.json` file) are located.
3.  Select an image you want to classify.
4.  The model will predict the class of the image and display the result.

## Project Structure

```
.
├── Leading.py
├── README.md
├── requirements.txt
└── Test_Data
    ├── class1
    │   ├── 1.jpg
    │   └── ...
    └── class2
        ├── 1.jpg
        └── ...
```

## Acknowledgements

*   [TensorFlow](https://www.tensorflow.org/)
*   [Pillow](https://python-pillow.org/)
*   [pyttsx3](https://pypi.org/project/pyttsx3/)
