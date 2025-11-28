#!/usr/bin/env python3
"""
Generate text variants for attribute groups using GPT-4o-mini.

This script takes the 18 attribute groups and generates 10 diverse natural language
descriptions for each using OpenAI's GPT-4o-mini model.

Usage:
    cd data_augment/
    # Create .env file with: OPENAI_API_KEY=your_key
    python generate_variants.py

Output:
    augmented_texts.json - 180 text descriptions (10 per group)
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# The 18 attribute groups from validation analysis
ATTRIBUTE_GROUPS = [
    "male, medium-pitched, measured speed",
    "male, low-pitched, measured speed",
    "female, high-pitched, measured speed",
    "female, medium-pitched, measured speed",
    "male, high-pitched, measured speed",
    "female, high-pitched, slow speed",
    "male, medium-pitched, fast speed",
    "male, high-pitched, slow speed",
    "male, low-pitched, fast speed",
    "male, medium-pitched, slow speed",
    "female, medium-pitched, fast speed",
    "female, low-pitched, measured speed",
    "male, low-pitched, slow speed",
    "female, medium-pitched, slow speed",
    "female, high-pitched, fast speed",
    "female, low-pitched, fast speed",
    "male, high-pitched, fast speed",
    "female, low-pitched, slow speed",
]

def generate_variants_for_group(client: OpenAI, attr_group: str, num_variants: int = 10) -> list[str]:
    """
    Generate diverse text variants for a single attribute group.

    Args:
        client: OpenAI client instance
        attr_group: Attribute group string (e.g., "male, medium-pitched, measured speed")
        num_variants: Number of variants to generate

    Returns:
        List of text variant strings
    """
    prompt = f"""Generate {num_variants} diverse natural language descriptions for a speaker with these attributes: {attr_group}

Make them varied in phrasing but semantically equivalent. Focus on being natural and conversational.

Examples of good variations:
- A man speaking at a moderate pace with average pitch
- Male voice with medium tone and steady delivery
- A gentleman with a balanced vocal range speaking at a measured rhythm

Return ONLY the {num_variants} descriptions, one per line, no numbering or extra text."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,  # Higher temperature for more diversity
    )

    # Parse response
    variants_text = response.choices[0].message.content.strip()
    variants = [line.strip() for line in variants_text.split('\n') if line.strip()]

    # Clean up any numbering that might have been added
    cleaned_variants = []
    for variant in variants:
        # Remove leading numbers like "1. ", "1) ", etc.
        cleaned = variant
        if cleaned and cleaned[0].isdigit():
            # Find first non-digit, non-punct character
            for i, char in enumerate(cleaned):
                if char.isalpha():
                    cleaned = cleaned[i:]
                    break
        cleaned_variants.append(cleaned)

    return cleaned_variants[:num_variants]  # Ensure exactly num_variants


def main():
    """Generate augmented text variants for all attribute groups."""

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment. Create .env file with your API key.")

    client = OpenAI(api_key=api_key)

    print("="*80)
    print("Generating Text Variants with GPT-4o-mini")
    print("="*80)
    print(f"\nAttribute groups: {len(ATTRIBUTE_GROUPS)}")
    print(f"Variants per group: 10")
    print(f"Total variants to generate: {len(ATTRIBUTE_GROUPS) * 10}\n")

    augmented_texts = {}

    for i, attr_group in enumerate(ATTRIBUTE_GROUPS, 1):
        print(f"[{i}/{len(ATTRIBUTE_GROUPS)}] Generating variants for: {attr_group}")

        try:
            variants = generate_variants_for_group(client, attr_group, num_variants=10)
            augmented_texts[attr_group] = variants

            print(f"  ✓ Generated {len(variants)} variants")
            print(f"    Example: \"{variants[0]}\"")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            raise

    # Save to JSON
    output_file = Path(__file__).parent / "augmented_texts.json"
    with open(output_file, 'w') as f:
        json.dump(augmented_texts, f, indent=2)

    print(f"\n" + "="*80)
    print(f"✓ Successfully generated {len(augmented_texts)} groups × 10 variants")
    print(f"✓ Saved to: {output_file}")
    print("="*80)

    # Print summary statistics
    total_variants = sum(len(variants) for variants in augmented_texts.values())
    print(f"\nSummary:")
    print(f"  Groups: {len(augmented_texts)}")
    print(f"  Total variants: {total_variants}")
    print(f"  Average per group: {total_variants / len(augmented_texts):.1f}")

    print("\nNext steps:")
    print("1. Review augmented_texts.json to verify quality")
    print("2. Transfer to cluster: scp augmented_texts.json psingh54@login.clsp.jhu.edu:/path/to/project/data_augment/")
    print("3. Update AUGMENTED_TEXTS path in run.sh")


if __name__ == "__main__":
    main()
