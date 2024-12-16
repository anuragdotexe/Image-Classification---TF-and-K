# **Image Classification Using TensorFlow and Keras**

This is a simple **image classification project** that uses a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10 dataset** into one of 10 categories.

---

## **Project Overview**
The CIFAR-10 dataset contains 60,000 32x32 color images categorized into 10 different classes:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

This project trains a CNN model using TensorFlow/Keras to recognize these categories and evaluate its performance.

---

## **Features**
- Loads and preprocesses the CIFAR-10 dataset.
- Builds and trains a Convolutional Neural Network (CNN).
- Visualizes training and validation accuracy.
- Tests the model on unseen images.
- Predicts the category of any uploaded image.

---

## **Technologies Used**
- **Python 3.7+**
- **TensorFlow/Keras**: For building and training the CNN.
- **Matplotlib**: For visualizing performance.
- **Google Colab**: For running the project in a cloud environment.

---

## **Setup Instructions**
1. Open **Google Colab** at [colab.research.google.com](https://colab.research.google.com/).
2. Create a new notebook and copy-paste the project code.
3. Install TensorFlow if needed:
   ```bash
   !pip install tensorflow
   ```
4. Run each code cell in sequence.

---

## **How It Works**
1. **Data Loading:**
   - The CIFAR-10 dataset is loaded using TensorFlow/Keras.
   - Images are normalized to a range of `[0, 1]`.

2. **Model Building:**
   - A Convolutional Neural Network (CNN) is created with 3 convolutional layers, followed by dense layers.

3. **Training:**
   - The model is trained on 50,000 images and validated on 10,000 images over 10 epochs.

4. **Evaluation:**
   - The trained model's accuracy is evaluated on test data.
   - A graph of training vs. validation accuracy is plotted.

5. **Prediction:**
   - The model predicts the class of new or unseen images.

---

## **Running Predictions**
1. Load an image for prediction (either from the test set or an external image).
2. Use the trained model to predict its category.
3. Output the category name (e.g., "Cat" or "Airplane").

Example code to predict a class:
```python
# Class labels
class_labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Predict the class
predicted_class = np.argmax(prediction)
print(f"Predicted Class: {class_labels[predicted_class]}")
```

---

## **Results**
- The model achieves ~70-80% accuracy on test data.
- It can correctly classify unseen images into one of the 10 categories.

---

## **Future Improvements**
- Add data augmentation to improve generalization.
- Use a more complex architecture for higher accuracy.
- Deploy the model using a simple web interface (e.g., Streamlit or Flask).

---

## **License**
This project is open-source and free to use under the MIT License.

---

Feel free to modify and enhance the project as per your requirements! 😊
