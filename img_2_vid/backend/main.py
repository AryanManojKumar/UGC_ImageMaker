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

load_dotenv()

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
        
        # Create agent and task
        script_agent = create_script_agent()
        
        task = Task(
            description=f"""Analyze the image at {image_path} and generate a creative
            {duration}-second video script. Return structured JSON with script_text,
            estimated_duration, and detailed segments for video generation.""",
            expected_output="JSON containing script_text, estimated_duration, and scene segments",
            agent=script_agent
        )
        
        crew = Crew(
            agents=[script_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        
        # Parse result
        script_data = json.loads(str(result))
        
        return JSONResponse(content={
            "success": True,
            "script": script_data,
            "image_path": str(image_path)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-audio-preview")
async def generate_audio_preview(
    script_text: str = Form(...),
    voice_settings: str = Form("{}")
):
    """Generate audio preview with user-specified settings"""
    try:
        audio_agent = create_audio_agent()
        
        params = json.dumps({
            "script_text": script_text,
            "voice_settings": json.loads(voice_settings)
        })
        
        task = Task(
            description=f"""Generate audio narration using these parameters: {params}
            Calculate exact duration and required video clips.""",
            expected_output="JSON with audio_path, duration, and required_clips",
            agent=audio_agent
        )
        
        crew = Crew(
            agents=[audio_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        audio_data = json.loads(str(result))
        
        return JSONResponse(content={
            "success": True,
            "audio": audio_data
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_video_production(
    job_id: str,
    image_path: str,
    script_data: dict,
    audio_data: dict
):
    """Background task for video production"""
    try:
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
            
            params = json.dumps({
                "image_path": image_path,
                "prompt": segment["prompt"],
                "clip_index": i + 1
            })
            
            task = Task(
                description=f"""Generate video clip {i+1}/{required_clips} with parameters: {params}
                Maintain visual consistency with previous clips.""",
                expected_output=f"Path to generated video clip {i+1}",
                agent=video_agent
            )
            
            crew = Crew(
                agents=[video_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            clip_path = str(crew.kickoff())
            clip_paths.append(clip_path)
            
            jobs[job_id]["progress"] = f"Generated clip {i+1}/{required_clips}"
        
        # Assemble video
        jobs[job_id]["status"] = "assembling"
        jobs[job_id]["progress"] = "Assembling video with audio..."
        
        assembly_params = json.dumps({
            "clip_paths": clip_paths,
            "audio_path": audio_data["audio_path"]
        })
        
        assembly_task = Task(
            description=f"""Assemble all video clips and attach audio: {assembly_params}
            Ensure perfect synchronization.""",
            expected_output="Path to assembled video with audio",
            agent=assembly_agent
        )
        
        assembly_crew = Crew(
            agents=[assembly_agent],
            tasks=[assembly_task],
            process=Process.sequential,
            verbose=True
        )
        
        assembled_video = str(assembly_crew.kickoff())
        
        # Apply lipsync
        jobs[job_id]["status"] = "lipsyncing"
        jobs[job_id]["progress"] = "Applying lip synchronization..."
        
        lipsync_params = json.dumps({
            "video_path": assembled_video,
            "audio_path": audio_data["audio_path"]
        })
        
        lipsync_task = Task(
            description=f"""Apply lip sync to video: {lipsync_params}
            Produce final video output.""",
            expected_output="Path to final lip-synced video",
            agent=lipsync_agent
        )
        
        lipsync_crew = Crew(
            agents=[lipsync_agent],
            tasks=[lipsync_task],
            process=Process.sequential,
            verbose=True
        )
        
        final_video = str(lipsync_crew.kickoff())
        
        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = "Video production complete!"
        jobs[job_id]["result"] = {
            "final_video_path": final_video,
            "duration": audio_data["duration"],
            "clips_generated": len(clip_paths)
        }
        
    except Exception as e:
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
