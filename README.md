Biometric Eye Authentication System

Project Overview
The Biometric Eye Authentication System is a deep learning-based application designed to authenticate users through facial recognition. By capturing images of users and training a convolutional neural network (CNN) model, the system can accurately identify individuals based on their unique facial features. This project aims to provide a secure and efficient method for user authentication, suitable for various applications such as access control and identity verification.

 Features
- User  Image Capture: Users can capture their facial images using a webcam, which are stored in a structured directory.
- Data Augmentation: The system applies various augmentation techniques to enhance the training dataset, improving model robustness and accuracy.
- Model Training: A CNN model is built and trained on the captured images, with the ability to save the trained model for future use.
- User  Management: Users can be added or deleted from the system, allowing for dynamic management of the authentication database.
- User  Authentication: The system can authenticate users in real-time by capturing an image and comparing it against the trained model.

 Technologies Used
- Python 3.x: The primary programming language for development.
- OpenCV: For image processing and capturing video from the webcam.
- TensorFlow: A deep learning framework used to build and train the CNN model.
- Keras: A high-level neural networks API for building and training models.
- NumPy: For numerical operations and handling arrays.
- JSON: For data serialization and saving class indices.
- Scikit-learn: For evaluation metrics such as classification report and F1 score.

 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/biometric-eye-authentication.git
   cd biometric-eye-authentication
   ```

2. Install the required packages:
   ```bash
   pip install opencv-python tensorflow numpy scikit-learn
   ```

 Usage
1. Capture Images: 
   - Run the `biometric_capture.py` script to capture images.
   - Enter a user name to create a directory for the captured images.
   - Press 'q' to stop capturing images.

2. Train Model:
   - After capturing images for one or more users, run the `biometric_capture.py` script and select the option to train the model.
   - The model will be trained on the captured images, and the trained model will be saved as `face_recognition_model.h5`.

3. Authenticate User:
   - Run the `authentication.py` script to authenticate a user.
   - The system will capture an image and predict the user based on the trained model.

4. Evaluate Model:
   - Run the `evaluation.py` script to evaluate the model's performance on the test dataset.
   - The script will output a classification report and the weighted F1 score.

5. Delete User:
   - To remove a user from the system, run the `biometric_capture.py` script and select the delete option, then enter the user name.

 Directory Structure
The project maintains a structured directory for storing user images and model files:
```
biometric-eye-authentication/
├── dataset/
│   ├── user1/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   ├── user2/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
├── face_recognition_model.h5
├── class_indices.json
├── biometric_capture.py
├── authentication.py
└── evaluation.py
```
 Model Architecture
The CNN model consists of the following layers:
- **Convolutional Layers**: Extract features from images using ReLU activation.
- **Max Pooling Layers**: Reduce dimensionality and retain essential features.
- **Batch Normalization Layers**: Improve training speed and stability.
- **Fully Connected Layers**: Include dropout for regularization to prevent overfitting.
- **Output Layer**: Uses softmax activation for multi-class classification.

 Evaluation
The evaluation script (evaluation.py) assesses the model's performance on the test dataset. It generates a classification report that includes precision, recall, and F1 score for each class. The weighted F1 score is also calculated to provide an overall measure of the model's accuracy across all classes. This evaluation helps in understanding the model's strengths and weaknesses, guiding further improvements.

Contributing
Contributions are highly encouraged! If you have suggestions for improvements or new features, please submit a pull request or open an issue. Ensure that your contributions align with the project's goals and maintain the code quality.





