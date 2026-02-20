import os
import json
import argparse
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader 

# Load the hidden variables from the .env file
load_dotenv()

def main():
    # 1. Set up the Command-Line Interface (CLI)
    parser = argparse.ArgumentParser(description="AI CV Screening Tool")
    parser.add_argument("--cv", type=str, default="cv.pdf", help="Path to the CV PDF file")
    parser.add_argument("--role", type=str, default="Backend Engineer", help="The job role to match against")
    args = parser.parse_args()

    print(f"\n🚀 Initializing AI CV Screener for the '{args.role}' role...")

    # Initialize the Gemini client securely
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # 2. Extract text from the requested PDF
    print(f"📄 Reading {args.cv}...")
    try:
        reader = PdfReader(args.cv)
        cv_text = "".join([page.extract_text() for page in reader.pages])
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{args.cv}'. Make sure the file exists.")
        return

    # 3. Create a strict prompt asking for JSON
    prompt = f"""
    You are an expert technical recruiter. Read this resume text and return ONLY a valid JSON object.
    Do not include markdown blocks like ```json. Just the raw JSON.
    
    The JSON should exactly match this structure:
    {{
      "candidate_name": "estimated name",
      "top_skills": ["skill 1", "skill 2", "skill 3"],
      "years_of_experience": "estimated years",
      "match_score_out_of_10": 8,
      "strengths": "One sentence on their biggest strength for this role",
      "weaknesses": "One sentence on what they are missing for this role"
    }}

    Target Role: {args.role}
    Resume Text:
    {cv_text}
    """

    print("🤖 AI is analyzing the CV...")
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )

    # 4. Token Tracking (The "Charles" Feature)
    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count
    print(f"📊 LLM Token Usage: {input_tokens} input | {output_tokens} output")

    # Clean up the response just in case the AI added markdown
    cleaned_response = response.text.strip().removeprefix("```json").removesuffix("```").strip()

    # 5. Save the result as a dynamic JSON file
    # (e.g., analysis_Backend_Engineer.json)
    output_filename = f"analysis_{args.role.replace(' ', '_')}.json"
    
    with open(output_filename, "w") as file:
        file.write(cleaned_response)

    print(f"✅ Success! AI analysis saved to '{output_filename}'\n")

if __name__ == "__main__":
    main()