import streamlit as st
import requests
import time
import json
from pathlib import Path

# Backend API URL
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="AI Video Production Studio",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 AI Video Production Studio")
st.markdown("Transform your images into professional narrated videos with AI")

# Initialize session state
if "script_data" not in st.session_state:
    st.session_state.script_data = None
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "image_path" not in st.session_state:
    st.session_state.image_path = None
if "job_id" not in st.session_state:
    st.session_state.job_id = None

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    video_duration = st.slider(
        "Target Video Duration (seconds)",
        min_value=14,
        max_value=70,
        value=30,
        step=7,
        help="Duration will be rounded to multiples of 7 seconds"
    )
    
    st.markdown("---")
    st.subheader("🎙️ Voice Settings")
    
    # ElevenLabs Configuration (Default)
    voice_model = st.selectbox(
        "ElevenLabs Model",
        [
            "eleven_turbo_v2_5",        # Fast, 3x faster
            "eleven_multilingual_v2"     # High quality, 29 languages
        ],
        index=0,  # Default to turbo
        help="Select ElevenLabs TTS model (via AIML API)"
    )
    
    voice_name = st.selectbox(
        "Voice",
        [
            "Rachel", "Nicole", "Aria", "Emily", "Jessica",  # Female voices first
            "Drew", "Clyde", "Paul", "Dave", "Roger",        # Male voices
            "Fin", "Sarah", "Antoni", "Laura", "Thomas", 
            "Charlie", "George", "Elli", "Callum", "Patrick", 
            "River", "Harry", "Liam", "Dorothy", "Josh", 
            "Arnold", "Charlotte", "Alice", "Matilda", "James", 
            "Joseph", "Will", "Jeremy", "Eric", "Michael",
            "Ethan", "Chris", "Gigi", "Freya", "Brian", 
            "Grace", "Daniel", "Lily", "Serena", "Adam", 
            "Bill", "Jessie", "Sam", "Glinda", "Giovanni", "Mimi"
        ],
        index=0,  # Default to Rachel
        help="Select ElevenLabs voice character"
    )
    
    # Advanced settings (collapsed by default)
    with st.expander("🎚️ Advanced Voice Settings"):
        stability = st.slider(
            "Stability",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Higher values = more consistent, Lower = more expressive"
        )
        
        similarity_boost = st.slider(
            "Similarity Boost",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.05,
            help="How closely to match the original voice"
        )
        
        style = st.slider(
            "Style",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Exaggeration of the voice style"
        )
        
        use_speaker_boost = st.checkbox(
            "Use Speaker Boost",
            value=True,
            help="Boost similarity to the speaker"
        )

# Main workflow
tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Upload Image",
    "📝 Script Generation",
    "🎙️ Audio Preview",
    "🎬 Video Production"
])

# Tab 1: Image Upload
with tab1:
    st.header("Upload Your Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Upload the image you want to animate"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.info(f"""
            **Image Details:**
            - Filename: {uploaded_file.name}
            - Size: {uploaded_file.size / 1024:.2f} KB
            - Target Duration: {video_duration} seconds
            - Required Clips: {video_duration // 7}
            """)
        
        if st.button("🚀 Analyze Image & Generate Script", type="primary"):
            with st.spinner("Analyzing image and generating script..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/api/analyze-image",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                        data={"duration": video_duration}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.script_data = result["script"]
                        st.session_state.image_path = result["image_path"]
                        st.success("✅ Script generated successfully!")
                        st.balloons()
                    else:
                        st.error(f"Error: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Tab 2: Script Review
with tab2:
    st.header("Review & Edit Script")
    
    if st.session_state.script_data:
        script = st.session_state.script_data
        
        st.subheader("📖 Generated Script")
        
        # Editable script text
        edited_script = st.text_area(
            "Script Text (you can edit this)",
            value=script["script_text"],
            height=200
        )
        
        script["script_text"] = edited_script
        
        st.markdown("---")
        st.subheader("🎬 Scene Breakdown")
        
        for idx, segment in enumerate(script["segments"]):
            with st.expander(f"Scene {idx + 1}: {segment['type'].upper()}", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Description:** {segment['description']}")
                    st.markdown(f"**Video Prompt:** {segment['prompt']}")
                
                with col2:
                    st.metric("Duration", f"{segment['duration']}s")
        
        st.markdown("---")
        
        if st.button("✅ Approve Script & Continue", type="primary"):
            st.session_state.script_data = script
            st.success("Script approved! Proceed to Audio Preview tab.")
            
    else:
        st.info("👈 Please upload an image and generate a script first.")

# Tab 3: Audio Preview
with tab3:
    st.header("Audio Preview & Configuration")
    
    if st.session_state.script_data:
        script = st.session_state.script_data
        
        st.subheader("🎙️ Voice Configuration")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Script Text:**")
            st.text_area(
                "Text to be narrated",
                value=script["script_text"],
                height=150,
                disabled=True
            )
        
        with col2:
            st.info(f"""
            **Audio Settings:**
            - Provider: ElevenLabs
            - Model: {voice_model}
            - Voice: {voice_name}
            - Estimated Duration: ~{script['estimated_duration']}s
            """)
        
        if st.button("🎵 Generate Audio Preview", type="primary"):
            with st.spinner("Generating audio with ElevenLabs..."):
                try:
                    # Prepare ElevenLabs voice settings
                    voice_settings = {
                        "model": voice_model,
                        "voice": voice_name,
                        "voice_settings": {
                            "stability": stability,
                            "similarity_boost": similarity_boost,
                            "style": style,
                            "use_speaker_boost": use_speaker_boost
                        }
                    }
                    
                    response = requests.post(
                        f"{API_BASE_URL}/api/generate-audio-preview",
                        data={
                            "script_text": script["script_text"],
                            "voice_settings": json.dumps(voice_settings)
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.audio_data = result["audio"]
                        
                        st.success("✅ Audio generated successfully!")
                        
                        # Display audio player
                        audio_file = st.session_state.audio_data["audio_path"]
                        st.audio(audio_file)
                        
                        st.info(f"""
                        **Audio Details:**
                        - Duration: {st.session_state.audio_data['duration']:.2f} seconds
                        - Required Video Clips: {st.session_state.audio_data['required_clips']}
                        - Clip Length: 7 seconds each
                        """)
                        
                    else:
                        st.error(f"Error: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        if st.session_state.audio_data:
            st.markdown("---")
            if st.button("✅ Approve Audio & Start Video Production", type="primary"):
                st.success("Audio approved! Proceed to Video Production tab.")
    else:
        st.info("👈 Please generate a script first.")

# Tab 4: Video Production
with tab4:
    st.header("Video Production")
    
    if st.session_state.script_data and st.session_state.audio_data:
        
        if st.session_state.job_id is None:
            st.subheader("🎬 Ready to Generate Video")
            
            st.info(f"""
            **Production Summary:**
            - Video Duration: {st.session_state.audio_data['duration']:.2f} seconds
            - Clips to Generate: {st.session_state.audio_data['required_clips']}
            - Estimated Time: ~{st.session_state.audio_data['required_clips'] * 2} minutes
            """)
            
            if st.button("🚀 Start Video Production", type="primary"):
                with st.spinner("Starting production..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/api/start-production",
                            data={
                                "image_path": st.session_state.image_path,
                                "script_data": json.dumps(st.session_state.script_data),
                                "audio_data": json.dumps(st.session_state.audio_data)
                            }
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.job_id = result["job_id"]
                            st.rerun()
                        else:
                            st.error(f"Error: {response.text}")
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        else:
            # Poll job status
            job_id = st.session_state.job_id
            
            status_placeholder = st.empty()
            progress_placeholder = st.empty()
            
            while True:
                try:
                    response = requests.get(f"{API_BASE_URL}/api/job-status/{job_id}")
                    
                    if response.status_code == 200:
                        job_status = response.json()
                        
                        status_placeholder.info(f"**Status:** {job_status['status']}")
                        progress_placeholder.text(job_status['progress'])
                        
                        if job_status['status'] == 'completed':
                            st.success("🎉 Video production complete!")
                            
                            result = job_status['result']
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                video_filename = Path(result['final_video_path']).name
                                st.video(result['final_video_path'])
                            
                            with col2:
                                st.metric("Duration", f"{result['duration']:.2f}s")
                                st.metric("Clips Generated", result['clips_generated'])
                                
                                st.download_button(
                                    label="📥 Download Video",
                                    data=open(result['final_video_path'], 'rb').read(),
                                    file_name=video_filename,
                                    mime="video/mp4"
                                )
                            
                            if st.button("🔄 Create New Video"):
                                st.session_state.clear()
                                st.rerun()
                            
                            break
                        
                        elif job_status['status'] == 'failed':
                            st.error(f"❌ Production failed: {job_status['error']}")
                            
                            if st.button("🔄 Try Again"):
                                st.session_state.job_id = None
                                st.rerun()
                            
                            break
                        
                        time.sleep(5)
                        st.rerun()
                        
                    else:
                        st.error("Error checking job status")
                        break
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    break
    
    else:
        st.info("👈 Please complete the previous steps first.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Powered by AIML API (GPT-4o Vision, ElevenLabs Turbo v2.5, Veo 3.1), Sync.so, and CrewAI</p>
</div>
""", unsafe_allow_html=True)