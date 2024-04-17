import cv2
import os
from tqdm import tqdm


def create_video(frames_dir, output_path, frame_rate=20):
    # Get the height and width from the first frame

    height, width, _ = cv2.imread(os.path.join(frames_dir, os.listdir(frames_dir)[0])).shape
    tmp = os.listdir(frames_dir)
    tmp = sorted(tmp, key=lambda x: int(x.split('.')[0].split('_')[1]))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    # Write each frame to the video file
    for f in tqdm(tmp, desc="Writting Frames"):
        frame = cv2.imread(os.path.join(frames_dir, f))
        video_writer.write(frame)

    # Release the video writer object
    video_writer.release()


if __name__ == '__main__':
    # # Example usage:
    # # Replace frames_list with your list of frames (each frame should be a NumPy array)
    frames_list = [cv2.imread(os.path.join(r"F:\Activity_Recognition_mmWare\Code\mmPose\Figure", f)) for f in
                   os.listdir(r"F:\Activity_Recognition_mmWare\Code\mmPose\Figure")]  # Your list of frames

    # output_video_path = 'output_video.avi'  # Output video file path
    #
    # # Make sure the frames_list is a list of NumPy arrays (images)
    # # The images should have the same height and width
    #
    # # Create the video
    output_video_path = 'mmMesh.avi'
    create_video(frames_list, output_video_path)
