import torch
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
import cv2
import os
from matplotlib import pyplot as plt
import gradio as gr
from gradio.themes.utils import colors, fonts, sizes
import re
from moviepy.editor import concatenate_videoclips, VideoFileClip
import fnmatch
from moviepy.editor import VideoFileClip
from PIL import Image
import numpy as np
from transformers import AutoProcessor, Blip2ForConditionalGeneration


def process_video(video_path, output_dir, fps_required):
    clip = VideoFileClip(video_path)
    mini_video_files = []
    images = []

    num_secs = 8//fps_required # Todo do not need to be int need to change for loop
    # Iterate over the video in 8-second intervals
    # for i in range(0, int(clip.duration), num_secs):
    #     start_time = i
    #     end_time = min(i + num_secs, clip.duration)

    #     mini_video = clip.subclip(start_time, end_time)

    #     mini_video = mini_video.set_duration(num_secs).set_fps(fps_required)

    #     # Define the file name for the mini video
    #     file_name = f"{output_dir}/mini_video_{i//num_secs}.mp4"

    #     # Write the mini video to a file
    #     mini_video.write_videofile(file_name, fps=1)

    #     # Append the file name to the list
    #     mini_video_files.append(file_name)
    
    mini_video_files = None
    frames = clip.iter_frames(fps=fps_required)
    for f in frames:
        images.append(f)
    print(f"I have {len(images)}.")
    return mini_video_files, images


def process_video_cv2(video_path, output_dir, fps_required):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Set the frames per second to fps_required
    video.set(cv2.CAP_PROP_FPS, fps_required)

    # Initialize frame count
    frame_count = 0

    # Initialize mini video count
    mini_video_count = 0
    mini_video_files = []

    while True:
        # Read the next frame from the video
        ret, frame = video.read()

        # If the frame was not read successfully, we're done
        if not ret:
            break

        # Save the frame to a file
        frame_file = os.path.join(output_dir, f'frame_{frame_count}.jpg')
        cv2.imwrite(frame_file, frame)

        # If we've collected 8 frames, create a mini video
        if frame_count % 8 == 7:
            mini_video_file = os.path.join(output_dir, f'mini_video_{mini_video_count}.mp4')

            # Create a VideoWriter object
            mini_video = cv2.VideoWriter(mini_video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps_required, (frame.shape[1], frame.shape[0]))

            # Write the frames to the mini video
            for i in range(8):
                frame_file = os.path.join(output_dir, f'frame_{frame_count - 7 + i}.jpg')
                frame = cv2.imread(frame_file)
                mini_video.write(frame)

                # Delete the frame file
                os.remove(frame_file)

            # Release the VideoWriter
            mini_video.release()

            mini_video_files.append(mini_video_file)

            # Increment the mini video count
            mini_video_count += 1

        # Increment the frame count
        frame_count += 1

    # Release the VideoCapture
    video.release()
    return mini_video_files

class ModelPredict:
    def __init__(self, mini_video_files, images, output_dir, video_path, fps_required):
        self.output_dir = output_dir
        self.video_path = video_path
        self.fps_required = fps_required
        self.device = 'cuda'
        self.device = torch.device(self.device)
        clip_type = {
            'video': 'LanguageBind_Video_FT',  # also LanguageBind_Video
            'audio': 'LanguageBind_Audio_FT',  # also LanguageBind_Audio
            'thermal': 'LanguageBind_Thermal',
            'image': 'LanguageBind_Image',
            'depth': 'LanguageBind_Depth',
        }

        self.model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
        self.model = self.model.to(self.device)
        self.model.eval()
        pretrained_ckpt = f'LanguageBind/LanguageBind_Image'
        self.tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
        self.modality_transform = {c: transform_dict[c](self.model.modality_config[c]) for c in clip_type.keys()}
        self.videos = mini_video_files
        self.images = images
        # Image processing
        # self.images = []
        # clip = VideoFileClip(self.video_path)
        # frames = clip.iter_frames(fps=fps_required)
        # for f in frames:
        #     self.images.append(f)
        # print(f'I have {len(self.images)} to find a similarity from')

        self.inputs = {
            'video': to_device(self.modality_transform['video'](self.videos), self.device),
            'image': to_device(self.modality_transform['image'](self.images), self.device),
        }        
  
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.vqa_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        self.vqa_model.to(self.device)

    def __predict_video__(self, text):
        language = [text]
        self.inputs['language'] = to_device(self.tokenizer(language, max_length=77, padding='max_length',
                                                    truncation=True, return_tensors='pt'), self.device)

        with torch.no_grad():
            embeddings = self.model(self.inputs)
        
            v = embeddings['video'] @ embeddings['language'].T
            s = torch.softmax(v, dim=0)
            s_flattened = s.view(-1)

            values_s, indices_s = torch.topk(s_flattened, 10)
            top_videos = [self.videos[i] for i in indices_s.tolist()]
            best_matched_video = top_videos[0]
            # print("Video x Text: \n",
            #       torch.softmax(embeddings['video'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())
            print(f"Top videos are: {top_videos}")
            return best_matched_video
    

    def __predict_image__(self, text):
        language = [text]
        self.inputs['language'] = to_device(self.tokenizer(language, max_length=77, padding='max_length',
                                                    truncation=True, return_tensors='pt'), self.device)

        with torch.no_grad():
            embeddings = self.model(self.inputs)
        
            v = embeddings['image'] @ embeddings['language'].T
            s = torch.softmax(v, dim=0)
            s_flattened = s.view(-1)
            values_s, indices_s = torch.topk(s_flattened, 10)
            top_images = [self.images[i] for i in indices_s.tolist()]
            top_images_saved = []
            # best_matched_image = top_images[0]
            answers = " "
            for i, im in enumerate(top_images):
                plt.imshow(im)
                plt.savefig(f"{self.output_dir}/frame{i}.jpg")  # For JPEG
                top_images_saved.append(f"{self.output_dir}/frame{i}.jpg")
                if 'how many' in text:
                    question = text
                else:
                    question = f'Is there a {text} in the image?'
                prompt = f"Question: {question} Answer:" 
                inputs = self.processor(Image.fromarray(im.astype('uint8')), text=prompt, return_tensors="pt").to(self.device, torch.float16)
                generated_ids = self.vqa_model.generate(**inputs, max_new_tokens=100)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                answers += f"{generated_text}\n" 
        
            return top_images_saved, answers


def chat_output_video(text):
    global model
    top_video_path = model.__predict_video__(text)
    return top_video_path

def chat_output_image_text(text):
    global model
    image_paths, output_text = model.__predict_image__(text)
    return image_paths, output_text

with gr.Blocks(title="Video Summarization",css="#chatbot {overflow:auto; height:500px;} #InputVideo {overflow:visible; height:320px;} footer {visibility: none}") as demo:


    with gr.Tab("Query - Video Output"):
        with gr.Row():
            #gr.Video(value='/scratch3/kat049/Ask-Anything/video_chat2/example/camera_long_6.mp4', show_label=False)
            chatbot = gr.Interface(
                fn=chat_output_video,
                inputs='text',
                outputs = [
                    gr.Video(label="Video Output"), 
                #outputs='video',
                ]
        )

    with gr.Tab("Query - Image + Text Output"):
        with gr.Row():
            #gr.Video(value='/scratch3/kat049/Ask-Anything/video_chat2/example/camera_long_6.mp4', show_label=False)
            chatbot = gr.Interface(
                fn=chat_output_image_text,
                inputs='text',
                outputs = [
                    gr.Gallery(columns=[5], rows=[2], label="Image Output", object_fit="contain", height="auto"), 
                    gr.Textbox(label="Text Output")
                #outputs='video',
                ]
        )


if __name__ == '__main__':
    output_dir = '/scratch3/kat049/Ask-Anything/video_chat2/example/out_30'
    #video_path = '/scratch3/kat049/Ask-Anything/video_chat2/example/camera_long_6.mp4'
    video_path = '/scratch3/kat049/tmp/out_30.mp4'
    fps_required = 1
    mini_video_files, images = process_video(video_path, output_dir, fps_required)

    mini_video_files = []
    for file in os.listdir(output_dir):
        if fnmatch.fnmatch(file, 'mini_video*.mp4'):
            mini_video_files.append(os.path.join(output_dir, file))

    mini_video_files.sort(key=lambda x: int(re.findall(r'\d+', x.split('/')[-1])[0]))

    global model
    model = ModelPredict(mini_video_files=mini_video_files, images=images, output_dir=output_dir, video_path=video_path, fps_required=fps_required)
    demo.queue()
    demo.launch()