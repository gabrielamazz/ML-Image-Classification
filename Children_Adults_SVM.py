import cv2
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import os
import matplotlib.pyplot as plt

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

#train_path = "C:\\Users\\Gabriela\\Desktop\\Anu 3 - sem 1\\TIA\\train"
#validate_path = "C:\\Users\\Gabriela\\Desktop\\Anu 3 - sem 1\\TIA\\validate"
#test_path = "C:\\Users\\Gabriela\\Desktop\\Anu 3 - sem 1\\TIA\\test"


# Incarcarea datelor
X_train, y_train = load_images_and_hog_features(train_path)
X_val, y_val = load_images_and_hog_features(validate_path)
X_test, y_test = load_images_and_hog_features(test_path)

# Crearea clasificatorului SVM cu un pipeline care include un StandardScaler
classifier = make_pipeline(StandardScaler(), svm.SVC(kernel='linear', probability=True))

# Antrenarea clasificatorului
classifier.fit(X_train, y_train)

# Evaluarea clasificatorului
y_pred_train = classifier.predict(X_train)
y_pred_val = classifier.predict(X_val)
y_pred_test = classifier.predict(X_test)

accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_val = accuracy_score(y_val, y_pred_val)
accuracy_test = accuracy_score(y_test, y_pred_test)

print("Training Accuracy:", accuracy_train)
print("Validation Accuracy:", accuracy_val)
print("Test Accuracy:", accuracy_test)

# Raportul de clasificare
print("\nClassification Report on Test Set:\n", classification_report(y_test, y_pred_test))

print("\nClassification Report on Training Set:\n", classification_report(y_train, y_pred_train))


#Grafic acuratete
accuracies = {'Train': accuracy_train, 'Validation': accuracy_val, 'Test': accuracy_test}

plt.bar(accuracies.keys(), accuracies.values())
plt.xlabel('Set de Date')
plt.ylabel('Acuratețe')
plt.title('Compararea Acurateților')
plt.show()


# Calculul matricei de confuzie ptr train
cm = confusion_matrix(y_train, y_pred_train)

# Afișarea matricei de confuzie
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()




# Calculul matricei de confuzie ptr test
cm = confusion_matrix(y_test, y_pred_test)

# Afișarea matricei de confuzie
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

