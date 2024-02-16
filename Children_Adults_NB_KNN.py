import cv2
from skimage.feature import hog
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt



# Functia de incarcare a datelor si extragerea caracteristicilor HOG este identica cu cea de la SVM.
# Detectorul Haar Cascade pentru fete
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Functia pentru incarcarea imaginilor, detectarea fetei si extragerea caracteristicilor HOG
def load_images_and_hog_features(folder_path):
    hog_features = []
    labels = []
    class_names = os.listdir(folder_path)

    for class_name in class_names:
        class_path = os.path.join(folder_path, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.endswith('.png'):
                    img_path = os.path.join(class_path, filename)
                    img = cv2.imread(img_path)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # Detectarea fetelor
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    for (x, y, w, h) in faces:
                        # Croirea imaginii pentru a include doar fata
                        face_roi = gray[y:y+h, x:x+w]
                        face_resized = cv2.resize(face_roi, (64, 64))

                        # Calcularea caracteristicilor HOG
                        features = hog(face_resized, orientations=9, pixels_per_cell=(8, 8),
                                       cells_per_block=(2, 2), visualize=False)
                        hog_features.append(features)
                        labels.append(class_name)
                        break  # Presupunem ca exista o singura fata relevanta per imagine

    return hog_features, labels

train_path = "C:\\Users\\Gabriela\\Desktop\\set de date\\train"
validate_path = "C:\\Users\\Gabriela\\Desktop\\set de date\\validate"
test_path = "C:\\Users\\Gabriela\\Desktop\\set de date\\test"


# Incarcarea datelor
X_train, y_train = load_images_and_hog_features(train_path)
X_val, y_val = load_images_and_hog_features(validate_path)
X_test, y_test = load_images_and_hog_features(test_path)

# Crearea si antrenarea clasificatorului KNN
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)
# Evaluarea clasificatorului KNN
y_pred_test_knn = knn_classifier.predict(X_test)
y_pred_train_knn = knn_classifier.predict(X_train)

# Crearea si antrenarea clasificatorului Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
# Evaluarea clasificatorului Naive Bayes
y_pred_test_nb = nb_classifier.predict(X_test)
y_pred_train_nb = nb_classifier.predict(X_train)

# Calculul acuratetii pentru fiecare clasificator
accuracy_train_knn = accuracy_score(y_train, y_pred_train_knn)
accuracy_test_knn = accuracy_score(y_test, y_pred_test_knn)

accuracy_train_nb = accuracy_score(y_train, y_pred_train_nb)
accuracy_test_nb = accuracy_score(y_test, y_pred_test_nb)



# Rapoarte de clasificare si matrici de confuzie pentru fiecare clasificator
print("KNN Classification Report:\n", classification_report(y_test, y_pred_test_knn))
print("Naive Bayes Classification Report:\n", classification_report(y_test, y_pred_test_nb))

# Graficul acuratetii pentru ambii algoritmi
accuracies = {
    'k-NN (Train)': accuracy_train_knn,
    'k-NN (Test)': accuracy_test_knn,
    'Naive Bayes (Train)': accuracy_train_nb,
    'Naive Bayes (Test)': accuracy_test_nb
}

# Crearea graficului cu bare
colors = ['blue', 'cyan', 'green', 'lightgreen']
plt.bar(accuracies.keys(), accuracies.values(), color=colors)
plt.xlabel('Classifier and Data Set')
plt.ylabel('Accuracy')
plt.title('Classifier Accuracies')
plt.ylim(0, 1)  # Acuratetea variaza intre 0 si 1
plt.show()

# Matricea de confuzie pentru KNN
cm_knn = confusion_matrix(y_test, y_pred_test_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=['Children', 'Adults'])
disp_knn.plot(cmap=plt.cm.Blues)
plt.title('Matricea de Confuzie - KNN')
plt.show()

# Matricea de confuzie pentru Naive Bayes
cm_nb = confusion_matrix(y_test, y_pred_test_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=['Children', 'Adults'])
disp_nb.plot(cmap=plt.cm.Blues)
plt.title('Matricea de Confuzie - Naive Bayes')
plt.show()
