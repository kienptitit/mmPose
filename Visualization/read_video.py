import cv2

# Replace 'your_video.avi' with the path to your .avi file
video_path = '/home/naver/Documents/Kien/mRI/mrPose/Visualization/subject18.avi'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video FPS: {fps}")
print(f"Video Resolution: {width} x {height}")

# Read and display each frame
while True:
    ret, frame = cap.read()

    # Break the loop if the video is over
    if not ret:
        break

    # Display the frame
    cv2.imshow('Video', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
