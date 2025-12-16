from crewai import Agent, LLM
from tools import (
    analyze_image_tool,
    generate_audio_tool,
    generate_video_clip_tool,
    assemble_video_tool,
    apply_lip_sync_tool
)
import os

# Configure LLM to use AIML API
llm = LLM(
    model="openai/gpt-4o",
    api_key=os.getenv("AIML_API_KEY"),
    base_url="https://api.aimlapi.com/v1/"
)

llm_11 = LLM(
    model="elevenlabs/v3_alpha",
    api_key=os.getenv("AIML_API_KEY"),
    base_url="https://api.aimlapi.com/v1/"
)

llm_v = LLM(
    model="google/veo-3.1-i2v",
    api_key=os.getenv("AIML_API_KEY"),
    base_url="https://api.aimlapi.com/v1/"
)




def create_script_agent():
    """Agent for image analysis and script generation"""
    return Agent(
        role="Vision Script Writer",
        goal="Analyze images and create compelling video scripts with precise timing and scene descriptions",
        backstory="""You are an expert scriptwriter with a background in visual storytelling
        and cinematography. You excel at analyzing images and crafting engaging narratives
        that translate perfectly to video format with proper pacing.""",
        tools=[analyze_image_tool],
        llm=llm,  # Add this
        verbose=True,
        allow_delegation=False
    )


def create_audio_agent():
    """Agent for audio generation and configuration"""
    return Agent(
        role="Audio Production Engineer",
        goal="Generate high-quality narration audio with precise timing calculations",
        backstory="""You are a professional audio engineer specializing in voice-over
        production and text-to-speech optimization. You ensure perfect audio quality
        and calculate exact timing requirements for video synchronization.""",
        tools=[generate_audio_tool],
        llm=llm_11,  # Add this
        verbose=True,
        allow_delegation=False
    )


def create_video_agent():
    """Agent for video clip generation"""
    return Agent(
        role="Video Production Specialist",
        goal="Generate visually consistent, high-quality video clips that maintain narrative flow",
        backstory="""You are a skilled video producer with expertise in AI-generated content.
        You understand visual continuity, shot composition, and how to maintain consistent
        style across multiple video clips. You craft detailed prompts that result in
        cohesive visual storytelling.""",
        tools=[generate_video_clip_tool],
        llm=llm_v,  # Add this
        verbose=True,
        allow_delegation=False
    )


def create_assembly_agent():
    """Agent for video assembly and post-production"""
    return Agent(
        role="Video Editor & Post-Production Specialist",
        goal="Seamlessly assemble video clips with perfect audio synchronization",
        backstory="""You are an expert video editor proficient in FFmpeg and multimedia
        processing. You ensure smooth transitions, perfect timing, and high-quality
        output in all your video assemblies.""",
        tools=[assemble_video_tool],
        verbose=True,
        allow_delegation=False
    )


def create_lipsync_agent():
    """Agent for lip synchronization"""
    return Agent(
        role="Lip Sync Technology Specialist",
        goal="Apply photorealistic lip synchronization to finalize video productions",
        backstory="""You are a specialist in advanced lip-sync technology, ensuring that
        character mouth movements perfectly match audio narration using state-of-the-art
        AI models. You deliver the final polished product.""",
        tools=[apply_lip_sync_tool],
        verbose=True,
        allow_delegation=False
    )
