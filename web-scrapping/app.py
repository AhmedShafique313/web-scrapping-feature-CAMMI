import os
import json
from dotenv import load_dotenv
from hyperbrowser import Hyperbrowser
from hyperbrowser.models import StartScrapeJobParams, ScrapeOptions
from huggingface_hub import InferenceClient

# Load environment variables (optional in Lambda, required for local testing)
load_dotenv()

# Initialize API clients
client_scraper = Hyperbrowser(api_key=os.getenv("HYPERBROWSER_API_KEY"))
client = InferenceClient(
    provider="fireworks-ai",
    api_key=os.getenv("HF_TOKEN")
)


def scrape_links(url: str):
    """Scrape all links from the given URL."""
    result = client_scraper.scrape.start_and_wait(
        StartScrapeJobParams(
            url=url,
            scrape_options=ScrapeOptions(formats=['links'], only_main_content=True)
        )
    )
    return result.data.links


def scrape_page_content(url: str):
    """Scrape main content in markdown format."""
    result = client_scraper.scrape.start_and_wait(
        StartScrapeJobParams(
            url=url,
            scrape_options=ScrapeOptions(formats=['markdown'], only_main_content=True)
        )
    )
    return result.data.markdown or ""


def llm_calling(prompt: str):
    """Call the LLM model with a given prompt."""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": str(prompt)}]
    )
    return response.choices[0].message.content.strip()


def lambda_handler(event, context):
    """
    Lambda handler for API Gateway with JSON request body.
    Expects:
    {
        "website": "https://example.com"
    }
    Returns JSON with structured company info.
    """
    try:
        # Parse incoming request
        body = json.loads(event.get("body", "{}"))
        website = body.get("website")

        if not website:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Missing 'website' in request body"})
            }

        # Step 1: scrape all links from the website
        links = scrape_links(website)
        links = [link for link in links if link.startswith(website)]

        # Step 2: scrape all page content
        all_content = ""
        for link in links:
            page_content = scrape_page_content(link)
            all_content += f"\n\n--- Page: {link} ---\n{page_content}"

        # Step 3: structure information
        prompt_structuring = (
            f"You are an expert information architect.\n"
            f"Convert the unstructured data below into structured information.\n"
            f"Do not remove any information, just present it in a structured format.\n\n"
            f"{all_content}"
        )
        structured_info = llm_calling(prompt_structuring)

        # Step 4: extract only marketing-relevant info
        prompt_relevancy = (
            f"You are an expert business and marketing analyst specializing in B2B strategy.\n"
            f"Review the following structured company information:\n{structured_info}\n\n"
            f"Extract only the information that is highly relevant for developing a B2B marketing strategy.\n"
            f"Discard irrelevant or redundant details.\n\n"
            f"Return the answer in the exact format below:\n\n"
            f"Objective:\n"
            f"Mission:\n"
            f"Vision:\n"
            f"Business Concept:\n"
            f"Target Market:\n"
            f"Value Proposition:\n"
            f"Business Name:\n"
            f"Products or Services they offer:\n"
            f"Who they currently sell to:\n"
            f"Top Business Goals:\n"
            f"Challenges:\n"
        )
        relevant_info = llm_calling(prompt_relevancy)

        # Step 5: finalize into clean professional format
        prompt_finalized = (
            f"You are a professional business document formatter.\n"
            f"Take the following input data:\n{relevant_info}\n\n"
            f"Remove any content between <think> and </think>.\n"
            f"Present the information in a clean, professional document format.\n\n"
            f"Required headings (exactly as written, in this order):\n"
            f"Objective:\n"
            f"Mission:\n"
            f"Vision:\n"
            f"Business Concept:\n"
            f"Target Market:\n"
            f"Value Proposition:\n"
            f"Business Name:\n"
            f"Products or Services they offer:\n"
            f"Who they currently sell to:\n"
            f"Top Business Goals:\n"
            f"Challenges:\n\n"
            f"Notes:\n"
            f"- Include only these headings and their corresponding information.\n"
            f"- If no information is available for a heading, write 'Not specified'.\n"
            f"- Do not use symbols like #, **, or bullet points.\n"
            f"- Do not add any extra commentary outside the headings.\n"
            f"- Maintain proper spacing and readability."
        )
        finalized_output = llm_calling(prompt_finalized)

        # Success response
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"structured_info": finalized_output})
        }

    except Exception as e:
        # Error response
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }
