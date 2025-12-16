"""
UGC Orchestrator Agent with True Multi-Tool Intelligence
Agent has 2 tools and orchestrates the workflow
"""
from crewai import Agent, Task, Crew, LLM
from prompt_variator_tool import PromptVariatorTool
from banana_tool_with_langsmith import BananaUGCTool
from dotenv import load_dotenv
import os
import langsmith
from langsmith import traceable

# Load environment variables
load_dotenv()

# Initialize LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "ugc-orchestrator")

@traceable(
    name="create_ugc_orchestrator_agent",
    tags=["agent-creation", "crewai", "multi-tool"],
    metadata={"model": "gpt-5-2025-08-07", "provider": "aiml-api", "num_tools": 2}
)
def create_ugc_orchestrator_agent():
    """
    Create a CrewAI agent with 2 tools:
    1. UGC Prompt Variator - generates 4 diverse prompts
    2. Banana UGC Image Generator - generates 1 image
    
    Agent orchestrates: Call tool 1 once, then call tool 2 four times
    """
    with langsmith.trace(
        name="initialize_tools",
        tags=["tool-initialization"]
    ) as tool_trace:
        prompt_variator = PromptVariatorTool()
        banana_tool = BananaUGCTool()
        tool_trace.outputs = {
            "tools": ["PromptVariatorTool", "BananaUGCTool"],
            "count": 2
        }

    with langsmith.trace(
        name="configure_llm",
        tags=["llm-configuration", "gpt-5"]
    ) as llm_trace:
        llm = LLM(
            model="openai/gpt-5-2025-08-07",
            api_key=os.getenv("AIML_API_KEY"),
            base_url="https://api.aimlapi.com/v1"
        )
        llm_trace.outputs = {"llm": "openai/gpt-5-2025-08-07"}

    agent = Agent(
        role="UGC Image Orchestrator",
        goal="Call UGC Prompt Variator once, then call Banana UGC Image Generator 4 times",
        backstory="""You orchestrate UGC image generation. 

Workflow:
1. Call "UGC Prompt Variator" once to get 4 prompts
2. When you see "NEXT STEP: Call Banana UGC Image Generator", immediately switch to using "Banana UGC Image Generator" tool
3. Call "Banana UGC Image Generator" 4 times with the 4 prompts you received
4. Report results

CRITICAL: After step 1 completes, you MUST use "Banana UGC Image Generator" tool for steps 2-5. Never call "UGC Prompt Variator" twice.""",
        tools=[prompt_variator, banana_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=7
    )

    return agent

@traceable(
    name="generate_ugc_with_orchestrator",
    tags=["multi-tool-orchestration", "ugc", "end-to-end"],
    metadata={"workflow": "orchestrated-multi-ugc", "expected_images": 4}
)
def generate_ugc_with_orchestrator(
    person_image_path: str,
    product_image_path: str,
    base_intent: str = None
):
    """
    Generate 4 diverse UGC images using intelligent agent orchestration.
    
    Args:
        person_image_path: Path to the person image
        product_image_path: Path to the product image
        base_intent: Base intent for image generation
    
    Returns:
        Agent result with confirmation of all 4 generated images
    """
    with langsmith.trace(
        name="validate_inputs",
        inputs={
            "person_image_path": person_image_path,
            "product_image_path": product_image_path,
            "base_intent": base_intent
        },
        tags=["validation"]
    ) as validate_trace:
        if not os.path.exists(person_image_path):
            error_msg = f"Person image not found: {person_image_path}"
            validate_trace.outputs = {"error": error_msg}
            return error_msg

        if not os.path.exists(product_image_path):
            error_msg = f"Product image not found: {product_image_path}"
            validate_trace.outputs = {"error": error_msg}
            return error_msg

        if not base_intent:
            base_intent = "A person showcasing a product in a natural, engaging way"

        validate_trace.outputs = {"status": "validated"}

    # Create orchestrator agent
    agent = create_ugc_orchestrator_agent()

    # Create task
    with langsmith.trace(
        name="create_orchestration_task",
        inputs={
            "person_image": person_image_path,
            "product_image": product_image_path,
            "base_intent": base_intent
        },
        tags=["task-creation"]
    ) as task_trace:
        task = Task(
            description=f"""Generate 4 UGC images. Execute these actions in order:

1. Call "UGC Prompt Variator" with base_intent="{base_intent}" to get 4 prompts

2. Call "Banana UGC Image Generator" with person_image_path={person_image_path}, product_image_path={product_image_path}, prompt=[first prompt], output_filename=generated_ugc_image_1.png

3. Call "Banana UGC Image Generator" with person_image_path={person_image_path}, product_image_path={product_image_path}, prompt=[second prompt], output_filename=generated_ugc_image_2.png

4. Call "Banana UGC Image Generator" with person_image_path={person_image_path}, product_image_path={product_image_path}, prompt=[third prompt], output_filename=generated_ugc_image_3.png

5. Call "Banana UGC Image Generator" with person_image_path={person_image_path}, product_image_path={product_image_path}, prompt=[fourth prompt], output_filename=generated_ugc_image_4.png

6. Report which images were generated successfully

Only call "UGC Prompt Variator" in step 1. Use "Banana UGC Image Generator" for steps 2-5.""",
            expected_output="List of successfully generated image files",
            agent=agent,
            human_input=False
        )
        task_trace.outputs = {"task": "orchestrated_multi_ugc_generation"}

    # Execute crew
    with langsmith.trace(
        name="execute_crew",
        tags=["crew-execution", "crewai", "orchestration"],
        metadata={"agents": 1, "tasks": 1, "tools": 2, "expected_images": 4}
    ) as crew_trace:
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True,
            max_iter=7,
            full_output=False
        )

        import time
        start_time = time.time()
        
        print("\n" + "="*60)
        print("Starting intelligent agent orchestration...")
        print("Agent will use 2 tools to generate 4 images")
        print("="*60 + "\n")
        
        result = crew.kickoff()
        execution_time = time.time() - start_time

        crew_trace.metadata["execution_time_seconds"] = round(execution_time, 2)
        crew_trace.outputs = {"result": str(result)}
        
        print("\n" + "="*60)
        print(f"Orchestration completed in {execution_time:.2f} seconds")
        print("="*60 + "\n")

    return result

if __name__ == "__main__":
    print("="*60)
    print("UGC Orchestrator Agent - Multi-Tool Intelligence")
    print("="*60)

    result = generate_ugc_with_orchestrator(
        person_image_path="person.jpg",
        product_image_path="product.jpg",
        base_intent="A happy person holding and showing off the product to the camera"
    )

    print("\n" + "="*60)
    print("Orchestration Result:")
    print("="*60)
    print(result)
