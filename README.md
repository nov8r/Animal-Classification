# Animal Image Classification with CNN

This project trains a Convolutional Neural Network (CNN) to classify images of animals into various categories. The notebook includes data preprocessing, model design, training, evaluation, and visualization of results.
It originated as a final project in collaboration with one of my classmates, Braydon Johnson, for my CSC 351 Machine Learning class. Post-completion of the class, I decided to make some minor improvements and polish it up a little bit more.
I may continue to revisit this project and try to further add to and improve my model.

Colab link: https://colab.research.google.com/drive/1omnEIpH496ARGgZ5im4oUCaMctFpYKAW?usp=sharing

## Dataset

The data we used in this project is the Animals-10 dataset from Kaggle.

Animals-10 Dataset Details:
- Contains ~26k Images
- 10 Different classes:
  - Dog, Horse, Elephant, Butterfly, Chicken, Cat, Cow, Sheep, Squirrel, and Spider
- All images in the dataset were gathered from Google Images.
- Dataset Link: https://www.kaggle.com/datasets/alessiocorrado99/animals10

Images are resized and augmented using torchvision transforms.

## Features and Tools Used

### Libraries:
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn** – Data Manipulation & Visualization
- **Scikit-Learn**, **PyTorch** – Model Building, Preprocessing, Evaluation
- **Google Colab** – Development Environment
- **kagglehub**, **os**, **random**, **PIL**, **TQDM** - Misc. Tools

### Model(s)

The CNN model is custom-built using PyTorch. Key architecture features:
- 4 convolutional layers
- Batch Normalization
- Max pooling
- Leaky ReLU activations
- Fully connected layers
- Dropout

CrossEntropyLoss is used as the loss function, and Adam optimizer is used for training.

We included three other pretrained models as well including, ResNet50, EfficientNetB0, and MobileNetV2

## Evaluation

- Accuracy, confusion matrix, and classification report are calculated on the test set.
- Training and validation losses are plotted over epochs.
- True and False predictions are visualized to diagnose model performance.

**Metrics used**:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Results

**Custom CNN Model**:  
**Loss and Accuracy over epochs**:  
![Loss and Accuracy over time](https://github.com/nov8r/Animal-Classification/blob/main/Images/Custom%20Model/Loss-Accuracy.png)  
**Confusion Matrix**:  
![Confusion Matrix](https://github.com/nov8r/Animal-Classification/blob/main/Images/Custom%20Model/Confusion%20Matrix.png)  
**Convolved Images**:  
![Convolved Images](https://github.com/nov8r/Animal-Classification/blob/main/Images/Custom%20Model/Convolutions.png)  
**True Predictions**:  
![True Predictions](https://github.com/nov8r/Animal-Classification/blob/main/Images/Custom%20Model/True%20Predictions.png)  
**False Predictions**:  
![False Predictions](https://github.com/nov8r/Animal-Classification/blob/main/Images/Custom%20Model/False%20Predictions.png)

**Custom Model Classification Report**:  
**Overall Accuracy: 0.7568**

| Class            | Precision | Recall | F1-Score |
|------------------|-----------|--------|----------|
| butterfly        | 0.8550    | 0.8066 | 0.8301   |
| cat              | 0.8276    | 0.4286 | 0.5647   |
| chicken          | 0.8592    | 0.7653 | 0.8095   |
| cow              | 0.7436    | 0.6170 | 0.6744   |
| dog              | 0.6845    | 0.8419 | 0.7551   |
| elephant         | 0.6198    | 0.8151 | 0.7041   |
| horse            | 0.7830    | 0.6996 | 0.7390   |
| sheep            | 0.6978    | 0.6978 | 0.6978   |
| spider           | 0.7711    | 0.9275 | 0.8421   |
| squirrel         | 0.8729    | 0.5508 | 0.6754   |
| **Macro avg**    | 0.7714    | 0.7150 | 0.7292   |
| **Weighted avg** | 0.7688    | 0.7568 | 0.7515   |

---

**ResNet50 Model**:  
**Loss and Accuracy over epochs**:  
![Loss and Accuracy over time](https://github.com/nov8r/Animal-Classification/blob/main/Images/ResNet%20Model/Loss-Accuracy.png)  
**Confusion Matrix**:  
![Confusion Matrix](https://github.com/nov8r/Animal-Classification/blob/main/Images/ResNet%20Model/Confusion%20Matrix.png)  
**Convolved Images**:  
![Convolved Images](https://github.com/nov8r/Animal-Classification/blob/main/Images/ResNet%20Model/Convolutions.png)  
**True Predictions**:  
![True Predictions](https://github.com/nov8r/Animal-Classification/blob/main/Images/ResNet%20Model/True%20Predictions.png)  
**False Predictions**:  
![False Predictions](https://github.com/nov8r/Animal-Classification/blob/main/Images/ResNet%20Model/False%20Predictions.png)

**ResNet50 Classification Report**  
**Overall Accuracy: 0.9151**

| Class            | Precision | Recall | F1-Score |
|------------------|-----------|--------|----------|
| butterfly        | 0.8995    | 0.9292 | 0.9142   |
| cat              | 0.8449    | 0.9405 | 0.8901   |
| chicken          | 0.9697    | 0.9260 | 0.9474   |
| cow              | 0.8925    | 0.8830 | 0.8877   |
| dog              | 0.9332    | 0.9179 | 0.9255   |
| elephant         | 0.8046    | 0.9589 | 0.8750   |
| horse            | 0.9740    | 0.8555 | 0.9109   |
| sheep            | 0.8366    | 0.9286 | 0.8802   |
| spider           | 0.9393    | 0.9296 | 0.9344   |
| squirrel         | 0.9483    | 0.8824 | 0.9141   |
| **Macro avg**    | 0.9043    | 0.9152 | 0.9080   |
| **Weighted avg** | 0.9187    | 0.9151 | 0.9156   |

---

**EfficientNetB0 Model**:  
**Loss and Accuracy over epochs**:  
![Loss and Accuracy over time](https://github.com/nov8r/Animal-Classification/blob/main/Images/Efficient%20Net/Loss-Accuracy.png)  
**Confusion Matrix**:  
![Confusion Matrix](https://github.com/nov8r/Animal-Classification/blob/main/Images/Efficient%20Net/Confusion%20Matrix.png)  
**Convolved Images**:  
![Convolved Images](https://github.com/nov8r/Animal-Classification/blob/main/Images/Efficient%20Net/Convolutions.png)  
**True Predictions**:  
![True Predictions](https://github.com/nov8r/Animal-Classification/blob/main/Images/Efficient%20Net/True%20Predictions.png)  
**False Predictions**:  
![False Predictions](https://github.com/nov8r/Animal-Classification/blob/main/Images/Efficient%20Net/False%20Predictions.png)

**EfficientNetB0 Classification Report**:  
**Overall Accuracy: 0.9410**

| Class            | Precision | Recall | F1-Score |
|------------------|-----------|--------|----------|
| butterfly        | 0.9528    | 0.9528 | 0.9528   |
| cat              | 0.9684    | 0.9107 | 0.9387   |
| chicken          | 0.9864    | 0.9357 | 0.9604   |
| cow              | 0.9477    | 0.8670 | 0.9056   |
| dog              | 0.9268    | 0.9363 | 0.9316   |
| elephant         | 0.9589    | 0.9589 | 0.9589   |
| horse            | 0.9007    | 0.9658 | 0.9321   |
| sheep            | 0.8418    | 0.9066 | 0.8730   |
| spider           | 0.9577    | 0.9855 | 0.9714   |
| squirrel         | 0.9718    | 0.9198 | 0.9451   |
| **Macro avg**    | 0.9413    | 0.9339 | 0.9370   |
| **Weighted avg** | 0.9423    | 0.9410 | 0.9411   |

---

**MobileNetV2 Model**:  
**Loss and Accuracy over epochs**:  
![Loss and Accuracy over time](https://github.com/nov8r/Animal-Classification/blob/main/Images/Mobile%20Net/Loss-Accuracy.png)  
**Confusion Matrix**:  
![Confusion Matrix](https://github.com/nov8r/Animal-Classification/blob/main/Images/Mobile%20Net/Confusion%20Matrix.png)  
**Convolved Images**:  
![Convolved Images](https://github.com/nov8r/Animal-Classification/blob/main/Images/Mobile%20Net/Convolutions.png)  
**True Predictions**:  
![True Predictions](https://github.com/nov8r/Animal-Classification/blob/main/Images/Mobile%20Net/True%20Predictions.png)  
**False Predictions**:  
![False Predictions](https://github.com/nov8r/Animal-Classification/blob/main/Images/Mobile%20Net/False%20Predictions.png)

**MobileNetV2 Classification Report**:  
**Overall Accuracy: 0.9242**

| Class            | Precision | Recall | F1-Score |
|------------------|-----------|--------|----------|
| butterfly        | 0.9431    | 0.9387 | 0.9409   |
| cat              | 0.7980    | 0.9643 | 0.8733   |
| chicken          | 0.9475    | 0.9293 | 0.9383   |
| cow              | 0.9314    | 0.8670 | 0.8981   |
| dog              | 0.9237    | 0.9199 | 0.9218   |
| elephant         | 0.9231    | 0.9041 | 0.9135   |
| horse            | 0.8877    | 0.9620 | 0.9234   |
| sheep            | 0.8994    | 0.8846 | 0.8920   |
| spider           | 0.9686    | 0.9565 | 0.9625   |
| squirrel         | 0.9695    | 0.8503 | 0.9060   |
| **Macro avg**    | 0.9192    | 0.9177 | 0.9170   |
| **Weighted avg** | 0.9268    | 0.9242 | 0.9245   |

---

# Final Thoughts

This project taught us a lot about CNNs and Residual Networks; We learned a lot throughout the whole process from the creation of our models to the evaluation.

Ultimately, our custom CNN implementation was too simple and underperformed significantly compared to the pretrained residual networks. It might be a good idea to go back and revisit our model and figure out ways to improve it.

We also tried to make our own custom implementation of a residual network, but it didn't workout well and we were short on time, but it may also be worth it to go back and revist that as well.

End Model Results:

| Model          | Accuracy | Precision | f1-Score | Recall  |
| :-----         | :------: | :-------: | :------: | :-----: |
| Custom         | 0.7568   | 0.7714    | 0.7292   | 0.7150  |
| ResNet50       | 0.9151   | 0.9043    | 0.9080   | 0.9152  |
| EfficientNetB2 | 0.9410   | 0.9413    | 0.9370   | 0.9339  |
| MobileNetV2    | 0.9242   | 0.9268    | 0.9170   | 0.9177  |

----

Note: I am aware that for both the Custom model and the ResNet50, the true predictions images contain false predictions. I will be working to fix that soon.

## Author

Project by **Ethan Posey**  
Original coursework: CSC 351 – Machine Learning
