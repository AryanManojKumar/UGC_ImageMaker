import os
import json
import shutil
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from crewai import Crew, Task, Process
from agents import (
    create_script_agent,
    create_audio_agent,
    create_video_agent,
    create_assembly_agent,
    create_lipsync_agent
)
import logging

load_dotenv()

# Configure logging to reduce verbosity
logging.getLogger("crewai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI(title="AI Video Production API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = Path("./temp_assets")
TEMP_DIR.mkdir(exist_ok=True)

# Job storage (in production, use database)
jobs = {}


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: str
    result: Optional[dict] = None
    error: Optional[str] = None


class VoiceSettings(BaseModel):
    model: str = "aura-asteria-en"
    stability: float = 0.5
    similarity_boost: float = 0.75


def extract_tool_result(result) -> str:
    """Extract clean result from CrewAI output"""
    if hasattr(result, 'raw'):
        return str(result.raw)
    return str(result)


def clean_json_response(text: str) -> str:
    """Clean markdown code blocks from JSON responses"""
    if "```json" in text:
        return text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()


@app.post("/api/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    duration: int = Form(30)
):
    """Analyze uploaded image and generate script"""
    try:
        # Save uploaded image
        image_path = TEMP_DIR / f"upload_{file.filename}"
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"📸 Analyzing image: {file.filename}")
        
        # Create agent and task
        script_agent = create_script_agent()
        
        task = Task(
            description=f"""Analyze the image at path: {str(image_path)}
            
Generate a creative {duration}-second video script using the 'Analyze Image and Generate Script' tool.

Pass these parameters as a JSON string:
{{"image_path": "{str(image_path)}", "duration": {duration}}}

The tool will return JSON with script_text, estimated_duration, and scene segments.""",
            expected_output="JSON containing script_text, estimated_duration, and scene segments",
            agent=script_agent
        )
        
        crew = Crew(
            agents=[script_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False  # Reduce verbosity
        )
        
        result = crew.kickoff()
        result_str = extract_tool_result(result)
        result_str = clean_json_response(result_str)
        
        script_data = json.loads(result_str)
        
        print(f"✅ Script generated: {len(script_data['script_text'])} characters")
        
        return JSONResponse(content={
            "success": True,
            "script": script_data,
            "image_path": str(image_path)
        })
        
    except Exception as e:
        print(f"❌ Script generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-audio-preview")
async def generate_audio_preview(
    script_text: str = Form(...),
    voice_settings: str = Form("{}")
):
    """Generate audio preview with user-specified settings"""
    try:
        print(f"🎙️ Generating audio narration...")
        
        audio_agent = create_audio_agent()
        
        voice_config = json.loads(voice_settings)
        
        task = Task(
            description=f"""Generate audio narration using the 'Generate Audio from Text' tool.

Pass these parameters as a JSON string:
{{"script_text": "{script_text[:100]}...", "voice_settings": {json.dumps(voice_config)}}}

The tool will return JSON with audio_path, duration, and required_clips.""",
            expected_output="JSON with audio_path, duration, and required_clips",
            agent=audio_agent
        )
        
        crew = Crew(
            agents=[audio_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False  # Reduce verbosity
        )
        
        result = crew.kickoff()
        result_str = extract_tool_result(result)
        result_str = clean_json_response(result_str)
        
        audio_data = json.loads(result_str)
        
        print(f"✅ Audio generated: {audio_data['duration']:.2f}s, {audio_data['required_clips']} clips needed")
        
        return JSONResponse(content={
            "success": True,
            "audio": audio_data
        })
        
    except Exception as e:
        print(f"❌ Audio generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def run_video_production(
    job_id: str,
    image_path: str,
    script_data: dict,
    audio_data: dict
):
    """Background task for video production"""
    try:
        print(f"\n🎬 Starting video production job: {job_id}")
        
        jobs[job_id]["status"] = "generating_videos"
        jobs[job_id]["progress"] = "Generating video clips..."
        
        # Create agents
        video_agent = create_video_agent()
        assembly_agent = create_assembly_agent()
        lipsync_agent = create_lipsync_agent()
        
        # Generate video clips
        segments = script_data["segments"]
        required_clips = audio_data["required_clips"]
        clip_paths = []
        
        for i in range(required_clips):
            # Determine which segment this clip belongs to
            if i == 0:
                segment = segments[0]  # intro
            elif i == required_clips - 1:
                segment = segments[-1]  # outro
            else:
                segment = segments[1]  # body
            
            print(f"\n🎥 Generating clip {i+1}/{required_clips}...")
            
            task = Task(
                description=f"""Generate video clip using 'Generate Video Clip with Veo 3.1' tool.

Pass these parameters as a JSON string:
{{"image_path": "{image_path}", "prompt": "{segment['prompt']}", "clip_index": {i + 1}}}

Return the path to the generated video clip.""",
                expected_output=f"Path to generated video clip {i+1}",
                agent=video_agent
            )
            
            crew = Crew(
                agents=[video_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False
            )
            
            result = crew.kickoff()
            clip_path = extract_tool_result(result).strip().strip('"\'')
            clip_paths.append(clip_path)
            
            jobs[job_id]["progress"] = f"Generated clip {i+1}/{required_clips}"
            print(f"✅ Clip {i+1} complete: {clip_path}")
        
        # Assemble video
        print(f"\n🔧 Assembling {len(clip_paths)} clips with audio...")
        jobs[job_id]["status"] = "assembling"
        jobs[job_id]["progress"] = "Assembling video with audio..."
        
        assembly_task = Task(
            description=f"""Assemble video using 'Assemble Video with FFmpeg' tool.

Pass these parameters as a JSON string:
{{"clip_paths": {json.dumps(clip_paths)}, "audio_path": "{audio_data['audio_path']}"}}

Concatenate all clips and attach the audio track.""",
            expected_output="Path to assembled video with audio",
            agent=assembly_agent
        )
        
        assembly_crew = Crew(
            agents=[assembly_agent],
            tasks=[assembly_task],
            process=Process.sequential,
            verbose=False
        )
        
        result = assembly_crew.kickoff()
        assembled_video = extract_tool_result(result).strip().strip('"\'')
        print(f"✅ Video assembled: {assembled_video}")
        
        # Apply lipsync
        print(f"\n💋 Applying lip synchronization...")
        jobs[job_id]["status"] = "lipsyncing"
        jobs[job_id]["progress"] = "Applying lip synchronization..."
        
        lipsync_task = Task(
            description=f"""Apply lip sync using 'Apply Lip Sync with Sync.so' tool.

Pass these parameters as a JSON string:
{{"video_path": "{assembled_video}", "audio_path": "{audio_data['audio_path']}"}}

Apply lip sync to produce the final video.""",
            expected_output="Path to final lip-synced video",
            agent=lipsync_agent
        )
        
        lipsync_crew = Crew(
            agents=[lipsync_agent],
            tasks=[lipsync_task],
            process=Process.sequential,
            verbose=False
        )
        
        result = lipsync_crew.kickoff()
        final_video = extract_tool_result(result).strip().strip('"\'')
        
        print(f"✅ Lip sync complete: {final_video}")
        print(f"\n🎉 Video production complete! Job: {job_id}")
        
        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = "Video production complete!"
        jobs[job_id]["result"] = {
            "final_video_path": final_video,
            "duration": audio_data["duration"],
            "clips_generated": len(clip_paths)
        }
        
    except Exception as e:
        print(f"❌ Production failed: {str(e)}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.post("/api/start-production")
async def start_production(
    background_tasks: BackgroundTasks,
    image_path: str = Form(...),
    script_data: str = Form(...),
    audio_data: str = Form(...)
):
    """Start video production workflow"""
    import uuid
    
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "job_id": job_id,
        "status": "started",
        "progress": "Initializing...",
        "result": None,
        "error": None
    }
    
    background_tasks.add_task(
        run_video_production,
        job_id,
        image_path,
        json.loads(script_data),
        json.loads(audio_data)
    )
    
    return JSONResponse(content={"job_id": job_id})


@app.get("/api/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JSONResponse(content=jobs[job_id])


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download generated file"""
    file_path = TEMP_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Video Production API...")
    print("📍 API will be available at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)