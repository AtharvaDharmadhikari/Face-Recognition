#Import KIvy Dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

# Build app and layout 
class CamApp(App):

    def build(self):
        # Main layout components 
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,.1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load tensorflow/keras model
        self.model = tf.keras.models.load_model("siamese_model")
        
        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout

    # Run continuously to get webcam feed
    def update(self, *args):

        # Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]

        # Flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Load image from file and conver to 100x100px
    def preprocess(self, file_path):
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image 
        img = tf.io.decode_jpeg(byte_img, channels=3)
        
        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, (100,100))
        # Scale image to be between 0 and 1 
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, axis=0)

        # 🔥 FORCE shape (safety)
        img = tf.ensure_shape(img, (1, 100, 100, 3))
        
        # Return image
        return img
        

    # Verification function to verify person
    def verify(self, *args):
        # Thresholds
        detection_threshold = 0.7
        verification_threshold = 0.4


        # Capture input image
        SAVE_PATH = os.path.join(
            'application_data', 'input_image', 'input_image.jpg'
        )

        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        verification_dir = os.path.join(
           'application_data', 'verification_images'
        )

        # 🔥 Get SavedModel inference function
        infer = self.model.signatures['serving_default']

        results = []

        for image in os.listdir(verification_dir):
            input_img = self.preprocess(SAVE_PATH)
            validation_img = self.preprocess(
                os.path.join(verification_dir, image)
            )

            # ✅ Correct inference for exported model
            output = infer(
                input_img=input_img,
                validation_img=validation_img
            )

            # Extract confidence score
            confidence = list(output.values())[0].numpy()[0][0]
            results.append(confidence)

        results = np.array(results)

        # Detection threshold
        detection = np.sum(results > detection_threshold)

        # Verification threshold
        verification = detection / len(results)
        verified = verification > verification_threshold

        # Update UI
        self.verification_label.text = (
            'Verified' if verified else 'Unverified'
        )

        # Logs
        Logger.info(f"Detection count: {detection}")
        Logger.info(f"Verification ratio: {verification}")
        Logger.info(f"Verified: {verified}")

        return results, verified




if __name__ == '__main__':
    CamApp().run()


