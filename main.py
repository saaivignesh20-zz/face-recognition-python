from cv2 import cv2
from PIL import Image
import tkinter
from tkinter import filedialog
import os
import glob
import face_recognition
import numpy as np
import string

def grabFrame():
    """
    This function grabs a frame from the camera.
    """

    try:
        ret, frame = camera.read()
        if not ret:
            return False
        # cv2.imshow("Face Recognizer", frame)
        return frame
    except:
        return False

def train_faces():
    global faces_encodings
    faces_encodings = []
    global faces_names
    faces_names = []
    cur_direc = os.getcwd()
    path = os.path.join(cur_direc, 'images/')
    list_of_files = [f for f in glob.glob(path+'*.jpg')]
    number_files = len(list_of_files)
    names = list_of_files.copy()

    for i in range(number_files):
        globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
        globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
        faces_encodings.append(globals()['image_encoding_{}'.format(i)])
        # Create array of known names
        names[i] = names[i].replace(cur_direc + "/images/", "").replace(".jpg", "").rstrip(string.digits).capitalize()
        faces_names.append(names[i])
    print(names)

def recognize_face_from_camera(frame = None):
    while True:
        frame = grabFrame()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces (faces_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance( faces_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = faces_names[best_match_index]
            face_names.append(name)
            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            #Input text label with a name below the face
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left, bottom + 24), font, 0.8, (0, 0, 255), 1)
        # Display the resulting image
        cv2.imshow('Face Recognizer', frame)
        # Hit 'ESC' on the keyboard to quit!
        if cv2.waitKey(1) == 27:
            break

def recognize_face_from_file(frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        print("Location: " + str(len(face_locations)))
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(faces_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance( faces_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = faces_names[best_match_index]
            face_names.append(name)
            # Display the results
            names_list = []
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
            
            names_list.append(name)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left, bottom + 24), font, 0.8, (0, 255, 255), 1)

        while True:
            # Display the resulting image
            cv2.imshow('Face Recognizer', frame)
            # Hit 'ESC' on the keyboard to quit!
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == 96:
                recognizeImageFile()

def recognizeImageFile():
    path = tkinter.filedialog.askopenfile()
    image = Image.open(path.name)
    image_arr = np.array(image)
    image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)
    # print(image_arr)
    recognize_face_from_file(image_arr) 
 
if __name__ == "__main__":
    """
    Driver Code
    """

    # global variables
    global camera
    global current_dir

    # get current working directory
    current_dir = os.getcwd()

    """
    # convert and save the image
    image_pic = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_pic.save("test.jpg")
    """

    # test routine
    # test_routine()
    train_faces()

    print("Menu\n\n1. Open Webcam\n2. Open Image\n\n")
    choice = int(input("Your Choice: "))

    if (choice == 1):
         # init camera
        camera = cv2.VideoCapture(-1)
        recognize_face_from_camera()
        # release camera
        camera.release()
        cv2.destroyAllWindows()
    else:
        root = tkinter.Tk()
        root.withdraw()
        path = tkinter.filedialog.askopenfile()
        image = Image.open(path.name)
        image_arr = np.array(image)
        image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)
        # print(image_arr)
        recognize_face_from_file(image_arr)