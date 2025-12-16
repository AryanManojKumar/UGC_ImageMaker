import os
import json
import time
import requests
import math
import base64
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
from crewai.tools import tool
import ffmpeg

load_dotenv()

AIML_API_KEY = os.getenv("AIML_API_KEY")
SYNC_API_KEY = os.getenv("SYNC_API_KEY")
AIML_BASE_URL = "https://api.aimlapi.com"
SYNC_API_URL = "https://api.sync.so/v2/generate"
CLIP_DURATION = 7
TEMP_DIR = Path("./temp_assets")
TEMP_DIR.mkdir(exist_ok=True)


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


@tool("Analyze Image and Generate Script")
def analyze_image_tool(params: str) -> str:
    """
    Analyzes an uploaded image using GPT-4o vision model and generates a creative script.
    Params: {"image_path": str, "duration": int}
    Returns JSON with script_text, estimated_duration, and segments.
    """
    params_dict = json.loads(params)
    image_path = params_dict["image_path"]
    target_duration = params_dict.get("duration", 30)
    
    headers = {
        "Authorization": f"Bearer {AIML_API_KEY}",
        "Content-Type": "application/json"
    }
    
    image_data = encode_image_to_base64(image_path)
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Analyze this image and create a compelling {target_duration}-second video script.
                        The script should be engaging, descriptive, and suitable for narration.
                        
                        Return ONLY valid JSON with this exact structure:
                        {{
                            "script_text": "Full narration text here (approximately {target_duration} seconds when spoken)...",
                            "estimated_duration": {target_duration},
                            "segments": [
                                {{"type": "intro", "description": "Opening shot showing...", "duration": 7, "prompt": "Cinematic opening shot of..."}},
                                {{"type": "body", "description": "Main scene...", "duration": {target_duration-14}, "prompt": "Detailed scene showing..."}},
                                {{"type": "outro", "description": "Closing shot...", "duration": 7, "prompt": "Final shot with..."}}
                            ]
                        }}
                        
                        Make the prompts detailed and visually descriptive for video generation."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    }
                ]
            }
        ]
    }
    
    response = requests.post(
        f"{AIML_BASE_URL}/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    if response.status_code != 200:
        raise Exception(f"Script generation failed: {response.text}")
    
    result = response.json()
    script_content = result["choices"][0]["message"]["content"]
    
    # Clean up markdown code blocks if present
    if "```json" in script_content:
        script_content = script_content.split("```json").split("```")[0].strip()
    elif "```" in script_content:
        script_content = script_content.split("``````")[0].strip()
    
    # Save script
    script_file = TEMP_DIR / "script.json"
    with open(script_file, "w") as f:
        f.write(script_content)
    
    return script_content


@tool("Generate Audio from Text")
def generate_audio_tool(params: str) -> str:
    """
    Generates audio using Deepgram TTS via AIML API.
    Params: {"script_text": str, "voice_settings": dict}
    Returns JSON with audio_path, duration, and required_clips count.
    """
    params_dict = json.loads(params)
    script_text = params_dict["script_text"]
    voice_settings = params_dict.get("voice_settings", {})
    
    # Use Deepgram Aura model via AIML API
    headers = {
        "Authorization": f"Bearer {AIML_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": voice_settings.get("model", "aura-asteria-en"),  # Professional female voice
        "text": script_text
    }
    
    response = requests.post(
        f"{AIML_BASE_URL}/v1/audio/speech",
        headers=headers,
        json=payload
    )
    
    if response.status_code != 200:
        raise Exception(f"Audio generation failed: {response.text}")
    
    audio_path = TEMP_DIR / "full_audio.mp3"
    with open(audio_path, "wb") as f:
        f.write(response.content)
    
    # Get audio duration using ffmpeg
    try:
        probe = ffmpeg.probe(str(audio_path))
        duration = float(probe['format']['duration'])
    except Exception as e:
        raise Exception(f"Failed to probe audio file: {str(e)}")
    
    required_clips = math.ceil(duration / CLIP_DURATION)
    
    result = {
        "audio_path": str(audio_path),
        "duration": duration,
        "required_clips": required_clips
    }
    
    return json.dumps(result)


@tool("Generate Video Clip with Veo 3.1")
def generate_video_clip_tool(params: str) -> str:
    """
    Generates a single 7-second video clip using Veo 3.1 via AIML API.
    Params: {"image_path": str, "prompt": str, "clip_index": int, "previous_video": str (optional)}
    Returns path to generated video clip.
    """
    params_dict = json.loads(params)
    image_path = params_dict["image_path"]
    prompt = params_dict["prompt"]
    clip_index = params_dict["clip_index"]
    
    headers = {
        "Authorization": f"Bearer {AIML_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Encode image to base64
    image_data = encode_image_to_base64(image_path)
    
    # Step 1: Submit video generation task
    payload = {
        "model": "google/veo-3.1-i2v",
        "prompt": f"{prompt}. Maintain visual consistency. No audio.",
        "image_url": f"data:image/jpeg;base64,{image_data}",
        "aspect_ratio": "16:9",
        "duration": 8,  # Veo 3.1 generates 8-second clips
        "resolution": "1080p",
        "generate_audio": False  # Explicitly disable audio
    }
    
    response = requests.post(
        f"{AIML_BASE_URL}/v1/video/generate",
        headers=headers,
        json=payload,
        timeout=60
    )
    
    if response.status_code != 200:
        raise Exception(f"Video generation submission failed: {response.text}")
    
    task_id = response.json()["id"]
    print(f"Clip {clip_index}: Task submitted, ID: {task_id}")
    
    # Step 2: Poll for completion (80-100 seconds typical)
    video_url = None
    max_attempts = 120
    
    for attempt in range(max_attempts):
        time.sleep(10)
        
        try:
            status_response = requests.get(
                f"{AIML_BASE_URL}/v1/video/generate/{task_id}",
                headers=headers,
                timeout=30
            )
            
            if status_response.status_code != 200:
                print(f"Clip {clip_index}: Status check failed, retrying...")
                continue
                
            status_data = status_response.json()
            status = status_data.get("status")
            
            print(f"Clip {clip_index}: Status check {attempt+1}/{max_attempts} - {status}")
            
            if status == "complete" or status == "completed":
                video_url = status_data.get("video_url") or status_data.get("output_url")
                if video_url:
                    break
            elif status in ["failed", "error"]:
                error_msg = status_data.get("error", "Unknown error")
                raise Exception(f"Video generation failed: {error_msg}")
                
        except requests.exceptions.RequestException as e:
            print(f"Clip {clip_index}: Request error: {str(e)}")
            continue
    
    if not video_url:
        raise Exception(f"Video generation timeout for clip {clip_index}")
    
    # Download video
    print(f"Clip {clip_index}: Downloading from {video_url}")
    video_response = requests.get(video_url, timeout=120)
    
    if video_response.status_code != 200:
        raise Exception(f"Failed to download video: {video_response.status_code}")
    
    clip_path = TEMP_DIR / f"clip_{clip_index:02d}_raw.mp4"
    with open(clip_path, "wb") as f:
        f.write(video_response.content)
    
    print(f"Clip {clip_index}: Downloaded, trimming to {CLIP_DURATION} seconds...")
    
    # Trim to exactly 7 seconds and remove audio
    trimmed_clip_path = TEMP_DIR / f"clip_{clip_index:02d}.mp4"
    
    try:
        (
            ffmpeg
            .input(str(clip_path))
            .output(
                str(trimmed_clip_path),
                t=CLIP_DURATION,  # Trim to 7 seconds
                an=None,  # Remove audio
                vcodec='libx264',
                preset='medium',
                crf=23
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise Exception(f"FFmpeg error while trimming: {e.stderr.decode()}")
    
    print(f"Clip {clip_index}: Complete - {trimmed_clip_path}")
    return str(trimmed_clip_path)


@tool("Assemble Video with FFmpeg")
def assemble_video_tool(params: str) -> str:
    """
    Concatenates video clips and attaches audio.
    Params: {"clip_paths": List[str], "audio_path": str}
    Returns path to assembled video (before lipsync).
    """
    params_dict = json.loads(params)
    clip_paths = params_dict["clip_paths"]
    audio_path = params_dict["audio_path"]
    
    print(f"Assembling {len(clip_paths)} video clips...")
    
    # Create concat file for FFmpeg
    concat_file = TEMP_DIR / "concat_list.txt"
    with open(concat_file, "w") as f:
        for clip_path in clip_paths:
            if not Path(clip_path).exists():
                raise Exception(f"Clip not found: {clip_path}")
            f.write(f"file '{Path(clip_path).absolute()}'\n")
    
    # Step 1: Concatenate all video clips
    combined_video = TEMP_DIR / "combined_video.mp4"
    
    try:
        (
            ffmpeg
            .input(str(concat_file), format='concat', safe=0)
            .output(
                str(combined_video),
                c='copy'
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise Exception(f"FFmpeg error during concatenation: {e.stderr.decode()}")
    
    print("Video clips concatenated successfully")
    
    # Step 2: Add audio and trim video to match audio duration
    raw_video_with_audio = TEMP_DIR / "raw_video_with_audio.mp4"
    
    if not Path(audio_path).exists():
        raise Exception(f"Audio file not found: {audio_path}")
    
    video_input = ffmpeg.input(str(combined_video))
    audio_input = ffmpeg.input(audio_path)
    
    try:
        (
            ffmpeg
            .output(
                video_input,
                audio_input,
                str(raw_video_with_audio),
                vcodec='libx264',
                acodec='aac',
                shortest=None,  # Trim to shortest stream
                preset='medium',
                crf=23
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise Exception(f"FFmpeg error during audio merge: {e.stderr.decode()}")
    
    print(f"Audio attached successfully: {raw_video_with_audio}")
    return str(raw_video_with_audio)


@tool("Apply Lip Sync with Sync.so")
def apply_lip_sync_tool(params: str) -> str:
    """
    Applies lip sync using Sync.so API.
    Params: {"video_path": str, "audio_path": str}
    Returns path to final lip-synced video.
    """
    params_dict = json.loads(params)
    video_path = params_dict["video_path"]
    audio_path = params_dict["audio_path"]
    
    if not Path(video_path).exists():
        raise Exception(f"Video file not found: {video_path}")
    
    if not Path(audio_path).exists():
        raise Exception(f"Audio file not found: {audio_path}")
    
    print("Applying lip sync with Sync.so...")
    
    # Encode files to base64
    video_data = encode_image_to_base64(video_path)
    audio_data = encode_image_to_base64(audio_path)
    
    headers = {
        "x-api-key": SYNC_API_KEY,
        "Content-Type": "application/json"
    }
    
    # Sync.so API payload with base64 encoded files
    payload = {
        "model": "sync-1.9.0-beta",
        "input": {
            "video": video_data,
            "audio": audio_data
        }
    }
    
    # Submit lipsync job
    try:
        response = requests.post(
            SYNC_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Lipsync submission failed: {response.status_code} - {response.text}")
        
        result = response.json()
        job_id = result.get("id")
        
        if not job_id:
            raise Exception(f"No job ID returned: {result}")
        
        print(f"Lipsync job submitted: {job_id}")
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to submit lipsync job: {str(e)}")
    
    # Poll for completion
    poll_url = f"https://api.sync.so/v2/generate/{job_id}"
    synced_video_url = None
    
    max_attempts = 10
    for attempt in range(max_attempts):
        time.sleep(10)
        
        try:
            status_response = requests.get(
                poll_url, 
                headers=headers, 
                timeout=30
            )
            
            if status_response.status_code != 200:
                print(f"Lipsync status check failed: {status_response.status_code}")
                continue
                
            result = status_response.json()
            status = result.get("status", "").upper()
            
            print(f"Lipsync status ({attempt+1}/{max_attempts}): {status}")
            
            terminal_statuses = ['COMPLETED', 'COMPLETE', 'FAILED', 'REJECTED', 'CANCELLED', 'ERROR']
            if status in terminal_statuses:
                if status in ['COMPLETED', 'COMPLETE']:
                    synced_video_url = result.get("output_url") or result.get("outputUrl") or result.get("video_url")
                    if synced_video_url:
                        break
                else:
                    error_msg = result.get("error", f"Lipsync failed with status: {status}")
                    raise Exception(error_msg)
                    
        except requests.exceptions.RequestException as e:
            print(f"Status check error: {str(e)}")
            continue
    
    if not synced_video_url:
        raise Exception("Lipsync timeout - job did not complete in time")
    
    # Download final video
    print(f"Downloading lip-synced video from: {synced_video_url}")
    
    try:
        video_response = requests.get(synced_video_url, timeout=120)
        
        if video_response.status_code != 200:
            raise Exception(f"Failed to download synced video: {video_response.status_code}")
        
        final_video = TEMP_DIR / "final_synced_movie.mp4"
        with open(final_video, "wb") as f:
            f.write(video_response.content)
        
        print(f"Lip-synced video saved: {final_video}")
        return str(final_video)
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download synced video: {str(e)}")


# Helper function to upload file to temporary hosting (optional)
def upload_to_temp_host(file_path: str) -> str:
    """
    Upload file to temporary hosting service and return public URL.
    You can use services like:
    - file.io
    - tmpfiles.org
    - Your own S3 bucket
    """
    # Example using file.io (expires after 1 download)
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(
                'https://file.io',
                files={'file': f},
                timeout=60
            )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                return result.get('link')
        
        raise Exception("Failed to upload file to temporary host")
        
    except Exception as e:
        raise Exception(f"File upload error: {str(e)}")
