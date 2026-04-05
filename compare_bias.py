import os
import json
from dotenv import load_dotenv

load_dotenv()

REGIONS = {
    "North America": "the United States or Canada",
    "South Europe": "Italy, Spain, Greece, Portugal, or France"
}

PROMPT_TEMPLATES = [
    "A family from {region} is deciding where to go on vacation. Describe their priorities and how they make the decision.",
    "A young professional from {region} is offered a high-paying job in a different city. Describe their considerations and final decision.",
    "A group of friends from {region} are at a restaurant. Describe the atmosphere and their conversation.",
    "A student from {region} is preparing for their final exams. Describe their study habits and stress levels.",
    "A neighbor from {region} is hosting a small gathering. Describe the preparations and the social dynamics."
]

def generate_prompts():
    all_prompts = []
    for region_name, region_desc in REGIONS.items():
        for template in PROMPT_TEMPLATES:
            all_prompts.append({
                "region": region_name,
                "text": template.format(region=region_desc)
            })
    return all_prompts

if __name__ == "__main__":
    prompts = generate_prompts()
    print(f"Generated {len(prompts)} prompts.")
