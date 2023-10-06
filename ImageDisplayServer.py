import pyigtl
import cv2
import numpy as np

# Create a PyIGTLink server
port = 18939
server = pyigtl.OpenIGTLinkServer(port=port)  # You can choose any available port

# Create a dictionary to store the last received image for each device name
image_data_dict = {}

# Start the server
server.start()
print(f'Image Display Server started on port {port}')

try:
    while True:
        # Accept incoming connections
        message_RGB = server.wait_for_message("RGB_Image", timeout=1)
        if message_RGB:
            cv2.imshow("RGB", message_RGB.image)
            
        message_Depth = server.wait_for_message("Depth_Image", timeout=1)
        if message_Depth:
            cv2.imshow("Depth", message_Depth.image)

        cv2.waitKey(1)

except KeyboardInterrupt:
    # Stop the server when Ctrl+C is pressed
    server.stop()
    cv2.destroyAllWindows()
