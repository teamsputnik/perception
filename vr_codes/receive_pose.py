import xr
import time
import xr
import numpy as np
import cv2

image = cv2.imread('C:/Users/15105/Desktop/example.jpeg')
print(image.shape)
height, width, _ = image.shape

# Calculate the halfway point to split the image vertically
half_width = width // 2

# Split the image into two parts (left and right)
left_image = image[:, :half_width]   # Left part of the image
right_image = image[:, half_width:]  # Right part of the image

# Check the result (Optional: display the images using OpenCV)

# Once XR_KHR_headless extension is ratified and adopted, we
# should be able to avoid the Window and frame stuff here.

with xr.InstanceObject(application_name="track_hmd") as instance, \
      xr.SystemObject(instance) as system, \
      xr.GlfwWindow(system) as window, \
      xr.SessionObject(system, graphics_binding=window.graphics_binding) as session:
    for _ in range(50):
        session.poll_xr_events()
        if session.state in (
                xr.SessionState.READY,
                xr.SessionState.SYNCHRONIZED,
                xr.SessionState.VISIBLE,
                xr.SessionState.FOCUSED,
        ):
            session.wait_frame()
            session.begin_frame()
            view_state, views = session.locate_views()
            print(views[xr.Eye.LEFT.value].pose, flush=True)
            print(views[xr.Eye.RIGHT.value].pose, flush=True)
            time.sleep(0.5)
            session.end_frame()