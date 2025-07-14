import markdown
import sys
import codecs

def convert_markdown_to_html(input_file, output_file):
    """
    Converts a Markdown file to an HTML file with syntax highlighting for code blocks.
    """
    try:
        with codecs.open(input_file, mode="r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    # Convert markdown to HTML using the 'extra' and 'codehilite' extensions
    # 'extra' adds features like tables, fenced code blocks, etc.
    # 'codehilite' adds syntax highlighting for code blocks
    html = markdown.markdown(text, extensions=['extra', 'codehilite'])

    # Basic HTML template with CSS for styling
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Markdown</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f9f9f9;
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: #111;
            }}
            code {{
                background-color: #eee;
                padding: 2px 4px;
                border-radius: 4px;
                font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
            }}
            pre {{
                background-color: #2d2d2d;
                color: #f8f8f2;
                padding: 16px;
                border-radius: 4px;
                overflow-x: auto;
            }}
            pre code {{
                background-color: transparent;
                padding: 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            blockquote {{
                border-left: 4px solid #ccc;
                padding-left: 16px;
                color: #666;
            }}
        </style>
    </head>
    <body>
        {html}
    </body>
    </html>
    """

    with codecs.open(output_file, "w", encoding="utf-8", errors="xmlcharrefreplace") as f:
        f.write(html_template)

    print(f"Successfully converted '{input_file}' to '{output_file}'")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python markdown_to_html.py <input_markdown_file> <output_html_file>")
        sys.exit(1)

    input_markdown_file = sys.argv[1]
    output_html_file = sys.argv[2]
    convert_markdown_to_html(input_markdown_file, output_html_file)
