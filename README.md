# Gesture-Recognition-with-Gemini

This project implements real-time gesture recognition using MediaPipe and OpenCV, and integrates Google Gemini for gesture interpretation and description. The system captures hand gestures, processes and sends the image to Gemini, from which a description of the image is sent back to the user.

The features consist of real-time gesture tracking, gesture detection and classification, automatic gesture capture, Google Gemini Integration, and an FPS counter for performance monitoring.

My main responsibilities consisted of engineering the pipeline between the user's gestures and Google Gemini. I added a framework such that when a gesture is detected, it signals the program to get ready to capture the image, but incorporates a delay such that the image is only captured when no gesture is detected. From there, the image is sent to Gemini for analysis. When a comprehensive description is generated, it is printed to the console, allowing for the user to ask follow-up questions to further their understanding. 

The intended effect of this project is to improve accessibility for visually impaired users, such that they can gain an understanding of their surroundings for obstacle avoidance and improve environmental navigation. 

The main code can be accessed via 20_gesture.py
