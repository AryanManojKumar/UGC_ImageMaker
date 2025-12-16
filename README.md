# UGC Image Orchestrator

AI-powered UGC (User Generated Content) image generator that creates 4 diverse product showcase images using CrewAI agents and Google's Nano Banana Pro Edit model.

## Features

- **Intelligent Agent Orchestration**: Uses CrewAI with GPT-5 to orchestrate multi-step image generation
- **Prompt Variation**: Automatically generates 4 diverse prompt variants from a base intent
- **Image Generation**: Creates realistic UGC-style images combining person and product photos
- **FastAPI Server**: RESTful API with chat-based interface
- **LangSmith Integration**: Optional tracing and monitoring for debugging

## Requirements

- Python 3.8+
- AI/ML API key (get from https://aimlapi.com)
- LangSmith API key (optional, for tracing)

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install crewai fastapi uvicorn requests python-dotenv langsmith openai pydantic
```

3. Create `.env` file from example:

```bash
cp .env.example .env
```

4. Add your API keys to `.env`:

```env
AIML_API_KEY=your_aiml_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here  # optional
```

## Usage

### Option 1: Run the FastAPI Server

Start the server:

```bash
python server_with_langsmith.py
```

The server will start at `http://localhost:8000`

**API Endpoints:**

- `POST /chat/ugc/upload` - Upload images and generate UGC content
- `GET /image/{filename}` - Retrieve generated images
- `GET /health` - Health check

**Example Request:**

```bash
curl -X POST "http://localhost:8000/chat/ugc/upload" \
  -F "message=person showcasing product in mountains" \
  -F "person_image=@person.jpg" \
  -F "product_image=@product.jpg"
```

### Option 2: Run Directly

```bash
python ugc_orchestrator_agent.py
```

Make sure you have `person.jpg` and `product.jpg` in the same directory.

## How It Works

1. **Prompt Generation**: The agent calls the UGC Prompt Variator tool to generate 4 diverse prompt variants
2. **Image Generation**: For each prompt, the agent calls the Banana UGC Image Generator tool
3. **Output**: 4 images are saved as `generated_ugc_image_1.png` through `generated_ugc_image_4.png`

## Models Used

- **Agent LLM**: `openai/gpt-5-2025-08-07` (via AI/ML API)
- **Image Generation**: `google/nano-banana-pro-edit` (Gemini 3 Pro Image Edit)

## Configuration

Edit `ugc_orchestrator_agent.py` to customize:

- `max_iter`: Maximum agent iterations (default: 7)
- `base_intent`: Default prompt intent
- Image paths and output filenames

## Troubleshooting

**Agent keeps calling prompt variator:**
- The agent should call it once, then switch to image generation
- Check that `max_iter` is set to at least 7

**API timeout errors:**
- Image generation can take 60-180 seconds per image
- Ensure stable internet connection
- Check AI/ML API status

**Images not generating:**
- Verify your AIML_API_KEY is valid
- Check that input images exist and are valid JPG/PNG files
- Review console output for specific error messages

## API Documentation

- AI/ML API: https://docs.aimlapi.com
- Nano Banana Pro Edit: https://docs.aimlapi.com/models/nano-banana-pro-edit
- CrewAI: https://docs.crewai.com

## License

MIT
