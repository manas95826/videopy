import requests
import io
import os
from PIL import Image
import moviepy.editor as mpy
from typing import List, Dict
from empire_chain.llms import GroqLLM
from empire_chain.podcast import GeneratePodcast
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GenerateVideo:
    def __init__(self):
        # Load API keys from environment variables
        self.hf_api_key = os.getenv('HF_API_KEY')  # Direct token
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        
        if not self.groq_api_key:
            raise ValueError("Missing GROQ_API_KEY in .env file.")
        
        self.API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
        self.headers = {"Authorization": f"Bearer {self.hf_api_key}"}
        
        # Initialize GroqLLM with the API key
        os.environ['GROQ_API_KEY'] = self.groq_api_key
        self.llm = GroqLLM()
        self.podcast_generator = GeneratePodcast()
        # Download required files for podcast generation
        self.podcast_generator.download_required_files()
        
    def _generate_prompts(self, topic: str, num_prompts: int = 10) -> List[str]:
        user_prompt = f"Generate {num_prompts} image prompts about {topic} that would work well in a video sequence."
        response = self.llm.generate(prompt=user_prompt)
        
        # Extract prompts from response
        prompts = []
        for line in response.split('\n'):
            line = line.strip()
            if line and not line.startswith(('#', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')):
                prompts.append(line)
        
        return prompts[:num_prompts]
        
    def _generate_image(self, prompt: str, output_path: str) -> str:
        def query(payload):
            """Generate a single image from a prompt"""
            response = requests.post(self.API_URL, headers=self.headers, json=payload)
            if response.status_code != 200:
                print(f"Full response content: {response.text}")
                print(f"Response headers: {response.headers}")
                raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
            return response.content
        
        try:
            image_bytes = query({"inputs": prompt})
            # Verify we got valid image data
            if not image_bytes.startswith(b'\x89PNG') and not image_bytes.startswith(b'\xFF\xD8\xFF'):
                print(f"Received content type: {type(image_bytes)}")
                print(f"First few bytes: {image_bytes[:100]}")
                raise ValueError(f"Invalid image data received from API: {image_bytes[:100]}")
            
            image = Image.open(io.BytesIO(image_bytes))
            image.save(output_path)
            return output_path
        except Exception as e:
            print(f"Error generating image for prompt: {prompt[:100]}...")
            print(f"Error details: {str(e)}")
            # Create a simple placeholder image
            placeholder = Image.new('RGB', (512, 512), color='gray')
            placeholder.save(output_path)
            return output_path
        
    def generate(self, topic: str, output_path: str = "final_video.mp4", fps: int = 24, num_prompts: int = 10) -> str:
        """Generate a video from a topic, handling both podcast and image generation"""
        # First generate the podcast audio
        print(f"Generating podcast for topic: {topic}")
        audio_path = self.podcast_generator.generate(topic=topic)
        
        # Generate prompts using GroqLLM
        print("Generating image prompts...")
        prompts = self._generate_prompts(topic, num_prompts)
        image_files = []
        
        # Generate images for each prompt
        print("Generating images...")
        for i, prompt in enumerate(prompts, 1):
            output_file = f"output_{i}.png"
            image_files.append(self._generate_image(prompt, output_file))
            print(f"Generated image {i} from prompt: {prompt[:100]}...")
            
        print("Creating video from images and audio...")
        # Create video
        audio_clip = mpy.AudioFileClip(audio_path)
        image_clips = [mpy.ImageClip(img).set_duration(audio_clip.duration / len(image_files)) 
                      for img in image_files]
        
        final_clip = mpy.concatenate_videoclips(image_clips)
        final_clip = final_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_path, fps=fps)
        
        # Clean up temporary image files
        for img_file in image_files:
            try:
                os.remove(img_file)
            except:
                pass
        
        return output_path 
