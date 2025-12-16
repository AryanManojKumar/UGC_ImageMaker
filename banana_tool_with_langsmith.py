import requests
import base64
import os
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import langsmith
from langsmith import traceable

class BananaUGCInput(BaseModel):
    """Input schema for BananaUGCTool."""
    person_image_path: str = Field(..., description="Path to the person image file")
    product_image_path: str = Field(..., description="Path to the product image file")
    prompt: str = Field(
        default="A person showcasing a product in a natural, engaging way",
        description="Description of how the person should showcase the product"
    )
    output_filename: str = Field(
        default="generated_ugc_image.png",
        description="Output filename for the generated image"
    )

class BananaUGCTool(BaseTool):
    name: str = "Banana UGC Image Generator"
    description: str = "Generates a UGC-style image where a person is showcasing a product using AI/ML API"
    args_schema: Type[BaseModel] = BananaUGCInput

    @traceable(
        name="banana_ugc_tool",
        tags=["image-generation", "banana-api", "ugc"],
        metadata={"model": "google/nano-banana-pro-edit", "provider": "aiml-api"}
    )
    def _run(
        self,
        person_image_path: str,
        product_image_path: str,
        prompt: str = "A person showcasing a product in a natural, engaging way",
        output_filename: str = "generated_ugc_image.png"
    ) -> str:
        """
        Generate UGC image using AI/ML API with google/nano-banana-pro model.
        Includes comprehensive LangSmith tracing for all steps.
        """
        api_key = os.getenv("AIML_API_KEY")
        if not api_key:
            return "Error: AIML_API_KEY not found in environment variables"

        # Read and encode images with tracing
        with langsmith.trace(
            name="encode_input_images",
            inputs={
                "person_image_path": person_image_path,
                "product_image_path": product_image_path
            },
            tags=["image-processing", "base64-encoding"],
            metadata={}
        ) as encode_trace:
            try:
                with open(person_image_path, "rb") as f:
                    person_image_data = f.read()
                    person_image_b64 = base64.b64encode(person_image_data).decode()

                with open(product_image_path, "rb") as f:
                    product_image_data = f.read()
                    product_image_b64 = base64.b64encode(product_image_data).decode()

                # Log image metadata
                encode_trace.metadata.update( {
                    "person_image_size": len(person_image_data),
                    "product_image_size": len(product_image_data),
                    "person_image_b64_length": len(person_image_b64),
                    "product_image_b64_length": len(product_image_b64)
                })
                encode_trace.outputs = {"status": "images_encoded"}

            except FileNotFoundError as e:
                error_msg = f"Error: Image file not found - {str(e)}"
                encode_trace.outputs = {"error": error_msg}
                return error_msg

        # AI/ML API endpoint
        url = "https://api.aimlapi.com/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Construct prompt for image-to-image composition
        full_prompt = (
            f"Combine these images to create a realistic UGC-style photo where the person "
            f"from the first image is naturally showcasing the product from the second image. "
            f"{prompt}. Keep the same person's face and identity, and the exact product appearance. "
            f"Make it look like authentic user-generated content."
        )

        # Use correct schema for google/nano-banana-pro-edit
        payload = {
            "model": "google/nano-banana-pro-edit",
            "prompt": full_prompt,
            "image_urls": [
                f"data:image/jpeg;base64,{person_image_b64}",
                f"data:image/jpeg;base64,{product_image_b64}"
            ],
            "aspect_ratio": "1:1",
            "resolution": "1K",
            "num_images": 1
        }

        # Call image generation API with detailed tracing
        with langsmith.trace(
            name="call_banana_api",
            inputs={
                "model": "google/nano-banana-pro-edit",
                "prompt": full_prompt,
                "num_images": 1
            },
            tags=["api-call", "image-generation", "aiml-api"],
            metadata={
                "provider": "AI/ML API",
                "endpoint": url,
                "model": "google/nano-banana-pro-edit"
            }
        ) as api_trace:
            try:
                import time
                start_time = time.time()

                print(f"\n🎨 Calling Banana API for {output_filename}...")
                print(f"   Model: {payload['model']}")
                print(f"   Prompt: {full_prompt[:100]}...")
                
                response = requests.post(url, json=payload, headers=headers, timeout=180)

                latency = time.time() - start_time

                # Log API call metadata
                api_trace.metadata.update({
                    "status_code": response.status_code,
                    "latency_seconds": round(latency, 2),
                    "response_size_bytes": len(response.content)
                })

                # Accept both 200 and 201 status codes
                if response.status_code not in [200, 201]:
                    error_msg = f"Error: API returned status {response.status_code}: {response.text[:500]}"
                    api_trace.outputs = {"error": error_msg, "status_code": response.status_code}
                    print(f"❌ API Error: {error_msg}")
                    return error_msg

                result = response.json()
                print(f"✅ API Success! Status: {response.status_code}")
                api_trace.outputs = {"status": "success", "result_keys": list(result.keys())}

                # Estimate cost (approximate pricing for nano-banana-pro-edit)
                # This is a placeholder - adjust based on actual pricing
                estimated_cost = 0.02  # $0.02 per generation (example)
                api_trace.metadata["estimated_cost_usd"] = estimated_cost

            except requests.exceptions.Timeout:
                error_msg = f"Error: API request timed out after 180 seconds"
                api_trace.outputs = {"error": error_msg}
                print(error_msg)
                return error_msg
            except requests.exceptions.RequestException as e:
                error_msg = f"Error calling AI/ML API: {str(e)[:200]}"
                api_trace.outputs = {"error": error_msg}
                print(error_msg)
                return error_msg

        # Save the generated image with tracing
        with langsmith.trace(
            name="save_generated_image",
            tags=["image-output", "file-save"],
            metadata={}
        ) as save_trace:
            if "data" in result and len(result["data"]) > 0:
                image_data = result["data"][0]
                output_path = output_filename

                # Check if it's a URL or base64
                if "url" in image_data:
                    # Download from URL
                    img_response = requests.get(image_data["url"])
                    img_response.raise_for_status()

                    with open(output_path, "wb") as f:
                        f.write(img_response.content)

                    save_trace.metadata.update( {
                        "method": "url_download",
                        "image_url": image_data["url"],
                        "output_path": output_path,
                        "image_size_bytes": len(img_response.content)
                    })
                    save_trace.outputs = {"output_path": output_path, "url": image_data["url"]}

                    return f"Success! UGC image generated and saved to {output_path}. Image URL: {image_data['url']}"

                elif "b64_json" in image_data:
                    # Decode base64
                    image_bytes = base64.b64decode(image_data["b64_json"])

                    with open(output_path, "wb") as f:
                        f.write(image_bytes)

                    save_trace.metadata.update({
                        "method": "base64_decode",
                        "output_path": output_path,
                        "image_size_bytes": len(image_bytes)
                    })
                    save_trace.outputs = {"output_path": output_path}

                    return f"Success! UGC image generated and saved to {output_path}"

                else:
                    error_msg = f"Error: Unexpected image format in response - {result}"
                    save_trace.outputs = {"error": error_msg}
                    return error_msg
            else:
                error_msg = f"Error: No image data in response - {result}"
                save_trace.outputs = {"error": error_msg}
                return error_msg
