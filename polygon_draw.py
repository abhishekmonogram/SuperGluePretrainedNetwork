import numpy as np
import cv2
import matplotlib.pyplot as plt
from pdb import set_trace as bp

# ============================================================================

CANVAS_SIZE = (600, 800)

FINAL_LINE_COLOR = (255, 255, 255)
WORKING_LINE_COLOR = (127, 127, 127)

# ============================================================================

class PolygonDrawer(object):
    def __init__(self, frame, window_name):
        self.window_name = window_name  # Name for our window
        self.frame = frame  # frame 1 after opening the webcam
        self.original_image = frame.copy()  # Capture the original image
        self.done = False  # Flag signaling we're done
        self.current = (0, 0)  # Current position, so we can draw the line-in-progress
        self.points = []  # List of points defining our polygon

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done:  # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update the current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at the current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.done = True

    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.window_name, np.zeros(CANVAS_SIZE, np.uint8))
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while not self.done:
            # This is our drawing loop, we continuously draw new images
            # and show them in the named window
            if len(self.points) > 0:
                # Draw all the current polygon segments
                cv2.polylines(self.frame, np.array([self.points]), False, FINAL_LINE_COLOR, 1)
                # And also show what the current segment would look like
                cv2.line(self.frame, self.points[-1], self.current, WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow(self.window_name, self.frame)
            # And wait 50ms before the next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27:  # ESC hit
                self.done = True

        # User finished entering the polygon points, so let's make the final drawing
        if len(self.points) > 0:
            # Create a mask for the polygon
            mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(self.points)], FINAL_LINE_COLOR)
            # Apply the mask to the original image
            result = cv2.bitwise_and(self.original_image, self.original_image, mask=mask)
            # Fill the remaining area outside the polygon with black
            outside_polygon = cv2.bitwise_not(mask)
            result[outside_polygon > 0] = 0
            self.frame = result

        # And show the result
        cv2.imshow(self.window_name, self.frame)
        # Waiting for the user to press any key
        cv2.waitKey()

        cv2.destroyWindow(self.window_name)
        return self.frame

# ============================================================================

if __name__ == "__main__":
    img = cv2.imread('assets/indoor_evaluation.png')
    pd = PolygonDrawer(img,"Polygon")
    image = pd.run()
    plt.imsave("polygon.png", image.astype(np.uint8))
    print("Polygon = %s" % pd.points)