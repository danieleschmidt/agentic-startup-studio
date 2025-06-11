from typing import Dict, Any


def generate_deck_content(deck_data: Dict[str, Any], template_path: str) -> str:
    """
    Generates Marp Markdown content by populating a template with deck_data.

    Args:
        deck_data: A dictionary containing information for the deck.
                   Expected keys include:
                   - "title" (str)
                   - "subtitle" (str)
                   - "contact_email" (str, optional)
                   - "sections" (List[Dict[str, str]]): A list of sections,
                     where each section is a dict with "title" and "content".
                     The content can be a string, which might contain markdown
                     bullet points (e.g., "* Point 1\n* Point 2").
                     The template expects sections to be directly addressable like
                     section1_title, section1_content, etc. For this basic version,
                     we'll map the first 3 sections if they exist.
        template_path: Path to the .marp template file.

    Returns:
        A string containing the populated Marp Markdown.
    """
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            template_content = f.read()
    except FileNotFoundError:
        return f"Error: Template file not found at {template_path}"
    except Exception as e:
        return f"Error reading template file: {e}"

    # Populate general placeholders
    content = template_content
    content = content.replace("{title}", deck_data.get("title", "Default Title"))
    default_subtitle = "Default Subtitle"
    content = content.replace("{subtitle}", deck_data.get("subtitle", default_subtitle))
    contact = deck_data.get("contact_email", "contact@example.com")
    content = content.replace("{contact_email}", contact)

    # Populate section placeholders
    # Assumes template has {sectionX_title} and {sectionX_content}
    sections = deck_data.get("sections", [])

    # A more dynamic template engine (like Jinja2) would handle
    # variable numbers of sections better.
    for i in range(1, 4):  # For section1, section2, section3
        title_ph = f"{{section{i}_title}}"
        content_ph = f"{{section{i}_content}}"

        if i <= len(sections):
            section = sections[i - 1]
            sec_title = section.get("title", f"Section {i} Title")
            sec_content = section.get("content", f"*No content for section {i}*")
            content = content.replace(title_ph, sec_title)
            content = content.replace(content_ph, sec_content)
        else:
            # If fewer sections provided, replace placeholders
            default_title = f"Section {i} (Not Provided)"
            default_content = "*Content not provided for this section.*"
            content = content.replace(title_ph, default_title)
            content = content.replace(content_ph, default_content)

    return content


if __name__ == "__main__":
    # Example Usage
    mock_deck_data = {
        "title": "My Awesome Startup Idea",
        "subtitle": "Revolutionizing the world, one slide at a time.",
        "contact_email": "info@awesomestartup.com",
        "sections": [
            {
                "title": "The Big Problem",
                "content": (
                    "* Current solutions are lacking.\n"
                    "* Users are frustrated.\n"
                    "* Huge market opportunity."
                ),
            },
            {
                "title": "Our Innovative Solution",
                "content": (
                    "* Introducing Product X!\n"
                    "* Key Features:\n"
                    "  * Feature A\n  * Feature B\n  * Feature C"
                ),
            },
            {
                "title": "Call to Action",
                "content": (
                    "* Invest in us!\n"
                    "* Join our team!\n"
                    "* Visit our website: awesomestartup.com"
                ),
            },
        ],
    }

    # Assume templates/deck_template.marp exists.
    # For robust path handling, use absolute paths or ensure script CWD.
    # Assumes 'templates' is subdir of project root or in PYTHONPATH.
    template_file = "templates/deck_template.marp"

    # Create a dummy template for direct execution if the real one is missing.
    import os

    if not os.path.exists(template_file):
        if not os.path.exists("templates"):
            os.makedirs("templates")
        dummy_template_content = """---
marp: true
---
# {title}
## {subtitle}
---
## {section1_title}
{section1_content}
---
## {section2_title}
{section2_content}
---
## {section3_title}
{section3_content}
---
Contact: {contact_email}
"""
        with open(template_file, "w", encoding="utf-8") as f:
            f.write(dummy_template_content)
        print(f"Dummy template created at {template_file} for __main__ execution.")

    generated_marp = generate_deck_content(mock_deck_data, template_file)
    print("\n--- Generated Marp Content ---")
    print(generated_marp)

    # Example with fewer sections
    mock_data_less = {
        "title": "Simpler Idea",
        "subtitle": "Less is More.",
        "sections": [{"title": "Core Idea", "content": "* One key point."}],
    }
    print("\n--- Generated Marp Content (Fewer Sections) ---")
    generated_marp_less = generate_deck_content(mock_data_less, template_file)
    print(generated_marp_less)
