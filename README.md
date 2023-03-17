# Face-Recognition
Using MTCNN is used to detect the location and size of the face in the image, extracting embeddings from the faces using the FaceNet model. Finally, SVM is used to  train a classifier on the facial embeddings extracted from the input images using the FaceNet model

## Detecting, embedding and training model
As we can see in the "Face Dectection&Recognition From Scratch" Jupyter notebook, I used colab to do the train things step by step, and saves "Faces_embedded_2class.npz" with two Numpy arrays to. The first array is 'EMBEDDED_X', which contains the embedded representations of the faces in the training set, and the second array is 'Y', which contains the corresponding labels for each face. The other is "svm_model1.pkl" contains the trained SVM model.

## Running realtime face recogniton
Using python file in the "Real_time_model" folder to run
```bash
python WC.py
```
## Testing
![ezgif-4-7bb35c23cd](https://user-images.githubusercontent.com/127477315/225806876-5ab6f9c1-17ea-490f-9231-483247022722.gif)
