"""
FastAPI Chat-based UGC Orchestrator with LangSmith Monitoring

Wraps the existing CrewAI UGC agent for continuous chat interaction
with comprehensive LangSmith tracing and monitoring
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import base64
import uuid
from datetime import datetime
from dotenv import load_dotenv

# LangSmith imports
from langsmith import Client, traceable
from langsmith.wrappers import wrap_openai
import langsmith

# Import UGC orchestrator agent (true multi-tool intelligence)
from ugc_orchestrator_agent import generate_ugc_with_orchestrator

# Load environment variables
load_dotenv()

# Initialize LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ugc-orchestrator"
# Make sure to set LANGCHAIN_API_KEY in your .env file

langsmith_client = Client()

app = FastAPI(title="UGC Orchestrator API", version="1.0.0")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store conversation history and generated images
conversations = {}
generated_images = {}

class ChatRequest(BaseModel):
    message: str
    person_image_path: Optional[str] = None
    product_image_path: Optional[str] = None
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    conversation_id: str
    assistant_message: str
    steps: List[Dict]
    generated_images: Optional[List[str]] = None  # Changed to list for 4 images
    timestamp: str
    trace_url: Optional[str] = None  # Added for LangSmith trace URL

@app.get("/")
async def serve_frontend():
    """Serve the HTML frontend"""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "Frontend not found"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "UGC Orchestrator API",
        "version": "1.0.0",
        "langsmith_enabled": os.getenv("LANGCHAIN_TRACING_V2") == "true"
    }

@traceable(
    name="chat_ugc_endpoint",
    tags=["fastapi", "ugc-generation"],
    metadata={"endpoint": "/chat/ugc"}
)
async def chat_ugc(request: ChatRequest):
    """Main chat endpoint - handles agent execution and image generation"""
    conversation_id = request.conversation_id or str(uuid.uuid4())

    if conversation_id not in conversations:
        conversations[conversation_id] = []

    conversations[conversation_id].append({
        "role": "user",
        "message": request.message,
        "timestamp": datetime.now().isoformat()
    })

    run_tree = langsmith.get_current_run_tree()
    trace_url = None

    try:
        # Build image metadata FIRST
        image_metadata = {
            "person_image_uploaded": request.person_image_path is not None,
            "product_image_uploaded": request.product_image_path is not None,
        }

        if request.person_image_path and os.path.exists(request.person_image_path):
            image_metadata["person_image_size"] = os.path.getsize(request.person_image_path)
        if request.product_image_path and os.path.exists(request.product_image_path):
            image_metadata["product_image_size"] = os.path.getsize(request.product_image_path)

        # Track uploaded images
        with langsmith.trace(
            name="process_uploaded_images",
            inputs={
                "person_image": request.person_image_path,
                "product_image": request.product_image_path
            },
            tags=["image-upload"],
            metadata=image_metadata
        ) as image_trace:
            image_trace.outputs = {"status": "images_processed"}

        # Check if user wants to generate images
        if request.person_image_path and request.product_image_path:
            # Use agent orchestrator to generate 4 images
            print(f"\n{'='*60}")
            print(f"Processing conversation: {conversation_id}")
            print(f"User message: {request.message}")
            print(f"{'='*60}\n")
            
            with langsmith.trace(
                name="agent_orchestration",
                inputs={"message": request.message, "conversation_id": conversation_id},
                tags=["agent-execution", "multi-tool"]
            ) as agent_trace:
                result = generate_ugc_with_orchestrator(
                    person_image_path=request.person_image_path,
                    product_image_path=request.product_image_path,
                    base_intent=request.message
                )
                agent_trace.outputs = {"result": str(result)}
            
            assistant_message = str(result)
        else:
            # No images provided, ask for them
            assistant_message = "Please upload both a person image and a product image to generate UGC images."
        
        steps = []

        # Handle multiple generated images (4 images)
        generated_image_list = []
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for i in range(1, 5):  # Check for 4 images
            original_filename = f"generated_ugc_image_{i}.png"
            if os.path.exists(original_filename):
                # Rename with conversation ID
                new_filename = f"ugc_{conversation_id}_{timestamp_str}_{i}.png"
                os.rename(original_filename, new_filename)
                generated_image_list.append(new_filename)
                
                # Build metadata
                image_size = os.path.getsize(new_filename)
                image_save_metadata = {
                    "filename": new_filename,
                    "size_bytes": image_size,
                    "conversation_id": conversation_id,
                    "variant_index": i
                }
                
                # Create trace with metadata
                with langsmith.trace(
                    name=f"save_generated_image_{i}",
                    tags=["image-output", "ugc-generation", f"variant-{i}"],
                    metadata=image_save_metadata
                ) as save_trace:
                    save_trace.outputs = {"image_path": new_filename}
        
        # Store all generated images for this conversation
        if generated_image_list:
            generated_images[conversation_id] = generated_image_list
            
            # Log feedback
            try:
                if run_tree and run_tree.id:
                    langsmith_client.create_feedback(
                        run_tree.id,
                        key="images_generated",
                        score=1.0,
                        comment=f"{len(generated_image_list)} images generated: {', '.join(generated_image_list)}"
                    )
            except Exception as e:
                print(f"Failed to log feedback: {e}")

        steps.append({
            "type": "agent_thinking",
            "description": "GPT-5 agent analyzed the request",
            "timestamp": datetime.now().isoformat()
        })

        if "Success" in assistant_message and "generated_ugc_image" in assistant_message:
            steps.append({
                "type": "tool_call",
                "tool": "Multi Banana UGC Image Generator",
                "description": f"Generated {len(generated_image_list)} diverse UGC images using nano-banana-pro-edit model",
                "timestamp": datetime.now().isoformat()
            })

        conversations[conversation_id].append({
            "role": "assistant",
            "message": assistant_message,
            "timestamp": datetime.now().isoformat()
        })

        # Get trace URL
        if run_tree and run_tree.id:
            try:
                tenant_id = langsmith_client._get_tenant_id()
                project_name = os.getenv("LANGCHAIN_PROJECT", "ugc-orchestrator")
                trace_url = f"https://smith.langchain.com/o/{tenant_id}/projects/p/{project_name}/r/{run_tree.id}"
            except Exception as e:
                print(f"Could not generate trace URL: {e}")
                trace_url = None

        return ChatResponse(
            conversation_id=conversation_id,
            assistant_message=assistant_message,
            steps=steps,
            generated_images=generated_image_list if generated_image_list else None,
            timestamp=datetime.now().isoformat(),
            trace_url=trace_url
        )

    except Exception as e:
        print(f"Error in chat_ugc: {str(e)}")
        if run_tree and run_tree.id:
            langsmith_client.create_feedback(
                run_tree.id,
                key="error",
                score=0.0,
                comment=f"Error: {str(e)}"
            )
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@traceable(name="chat_ugc_upload_endpoint", tags=["fastapi", "file-upload"])
async def chat_ugc_with_upload(
    message: str = Form(...),
    person_image: Optional[UploadFile] = File(None),
    product_image: Optional[UploadFile] = File(None),
    conversation_id: Optional[str] = Form(None)
):
    """Upload wrapper - just saves files and calls chat_ugc()"""
    person_image_path = None
    product_image_path = None

    # Prepare metadata
    upload_metadata = {}

    if person_image:
        person_image_path = f"temp_person_{uuid.uuid4()}.jpg"
        content = await person_image.read()
        with open(person_image_path, "wb") as f:
            f.write(content)
        upload_metadata["person_image"] = {
            "filename": person_image.filename,
            "size": len(content),
            "path": person_image_path
        }

    if product_image:
        product_image_path = f"temp_product_{uuid.uuid4()}.jpg"
        content = await product_image.read()
        with open(product_image_path, "wb") as f:
            f.write(content)
        upload_metadata["product_image"] = {
            "filename": product_image.filename,
            "size": len(content),
            "path": product_image_path
        }

    # Log upload
    with langsmith.trace(
        name="save_uploaded_files", 
        tags=["file-upload"],
        metadata=upload_metadata
    ) as upload_trace:
        upload_trace.outputs = {"status": "files_saved"}

    # Create request
    request = ChatRequest(
        message=message,
        person_image_path=person_image_path,
        product_image_path=product_image_path,
        conversation_id=conversation_id
    )

    # ✅ JUST CALL chat_ugc() - it handles everything else
    response = await chat_ugc(request)

    # Clean up temp files
    if person_image_path and os.path.exists(person_image_path):
        os.remove(person_image_path)
    if product_image_path and os.path.exists(product_image_path):
        os.remove(product_image_path)

    # ✅ NO IMAGE HANDLING HERE - just return the response
    return response



@app.post("/chat/ugc/upload", response_model=ChatResponse)
async def chat_ugc_upload_endpoint(
    message: str = Form(...),
    person_image: Optional[UploadFile] = File(None),
    product_image: Optional[UploadFile] = File(None),
    conversation_id: Optional[str] = Form(None)
):
    """Wrapper endpoint"""
    return await chat_ugc_with_upload(message, person_image, product_image, conversation_id)

@app.get("/image/{filename}")
async def get_image(filename: str):
    """
    Retrieve a generated image by filename
    """
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(filename, media_type="image/png")

@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    Retrieve conversation history
    """
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "conversation_id": conversation_id,
        "messages": conversations[conversation_id],
        "generated_image": generated_images.get(conversation_id)
    }

@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete conversation history
    """
    if conversation_id in conversations:
        del conversations[conversation_id]

    if conversation_id in generated_images:
        image_file = generated_images[conversation_id]
        if os.path.exists(image_file):
            os.remove(image_file)
        del generated_images[conversation_id]

    return {"status": "deleted", "conversation_id": conversation_id}

@app.post("/feedback/{conversation_id}")
async def submit_feedback(
    conversation_id: str,
    feedback_type: str,  # "thumbs_up" or "thumbs_down"
    run_id: Optional[str] = None,
    comment: Optional[str] = None
):
    """
    Submit user feedback for a conversation
    """
    try:
        if not run_id:
            return {"status": "error", "message": "run_id required"}

        score = 1.0 if feedback_type == "thumbs_up" else 0.0

        langsmith_client.create_feedback(
            run_id,
            key=feedback_type,
            score=score,
            comment=comment
        )

        return {
            "status": "success",
            "conversation_id": conversation_id,
            "feedback_type": feedback_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("🚀 Starting UGC Orchestrator API Server")
    print("="*60)
    print("📡 Endpoint: http://localhost:8000/chat/ugc")
    print("📚 Docs: http://localhost:8000/docs")
    print("🔍 LangSmith Tracing:", os.getenv("LANGCHAIN_TRACING_V2", "false"))
    print("📊 Project:", os.getenv("LANGCHAIN_PROJECT", "ugc-orchestrator"))
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)


