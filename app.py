import gradio as gr
from playwright.async_api import async_playwright, Page
from PIL import Image
from io import BytesIO
from anthropic import Anthropic, TextEvent
from dotenv import load_dotenv
import os
from typing import Literal
import time
from base64 import b64encode
import sys

load_dotenv()
# check for ANTHROPIC_API_KEY
if os.getenv("ANTHROPIC_API_KEY") is None:
    raise ValueError(
        "ANTHROPIC_API_KEY is not set, set it in .env or export it in your environment"
    )

anthropic = Anthropic()
model = "claude-3-5-sonnet-20240620"


def prepare_playwright_if_needed():
    # on linux install chromium with deps
    if sys.platform.startswith("linux"):
        os.system("playwright install chromium --with-deps")
    else:
        os.system("playwright install chromium")


prepare_playwright_if_needed()


def apply_tailwind(content):
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Document</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body>
        {content}
    </body>
    </html>
    """


system_prompt = f"""
You are a helpful assistant that generates HTML and CSS.

You directly output HTML with tailwind css classes, and nothing else (no markdown, no other text, etc).

You are not able to insert images from the internet, but you can still generate an <img> tag with an appropriate alt tag (leave out the src, we will provide that).

Assume that the content is being inserted into a template like this:
{apply_tailwind("your html here")}
"""

improve_prompt = """
Given the current draft of the webpage you generated for me as HTML and the screenshot of it rendered, improve the HTML to look nicer.
"""


def stream_initial(prompt):
    with anthropic.messages.stream(
        model=model,
        max_tokens=2000,
        system=system_prompt,
        messages=[
            {"role": "user", "content": prompt},
        ],
    ) as stream:
        for message in stream:
            if isinstance(message, TextEvent):
                yield message.text


def format_image(image: bytes, media_type: Literal["image/png", "image/jpeg"]):
    image_base64 = b64encode(image).decode("utf-8")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": image_base64,
        },
    }


def stream_with_visual_feedback(prompt, history: list[tuple[str, bytes]]):
    """
    history is a list of tuples of (content, image) corresponding to iterations of generation and rendering
    """
    print(f"History has {len(history)} images")

    messages = [
        {"role": "user", "content": prompt},
        *[
            item
            for content, image_bytes in history
            for item in [
                {
                    "role": "assistant",
                    "content": content,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Here is a screenshot of the above HTML code rendered in a browser:",
                        },
                        format_image(image_bytes, "image/png"),
                        {
                            "type": "text",
                            "text": improve_prompt,
                        },
                    ],
                },
            ]
        ],
    ]

    with anthropic.messages.stream(
        model=model,
        max_tokens=2000,
        system=system_prompt,
        messages=messages,
    ) as stream:
        for message in stream:
            if isinstance(message, TextEvent):
                yield message.text


async def render_html(page: Page, content: str):
    start_time = t()
    await page.set_content(content)
    # weird, can we set scale to 2.0 directly instead of "device", ie whatever server this is running on?
    image_bytes = await page.screenshot(type="png", scale="device", full_page=True)
    return image_bytes, t() - start_time


def t():
    return time.perf_counter()


def apply_template(content, template):
    if template == "tailwind":
        return apply_tailwind(content)
    return content


def to_pil(image_bytes: bytes):
    return Image.open(BytesIO(image_bytes))


async def generate_with_visual_feedback(
    prompt,
    template,
    resolution: str = "512",
    num_iterations: int = 1,
):
    render_every = 0.25
    resolution = {"512": (512, 512), "1024": (1024, 1024)}[resolution]
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(
            viewport={"width": resolution[0], "height": resolution[1]}
        )
        last_yield = t()
        history = []
        for i in range(num_iterations):
            stream = (
                stream_initial(prompt)
                if i == 0
                else stream_with_visual_feedback(prompt, history)
            )
            content = ""
            for chunk in stream:
                content = content + chunk
                current_time = t()
                if current_time - last_yield >= render_every:
                    image_bytes, render_time = await render_html(
                        page, apply_template(content, template)
                    )
                    yield to_pil(image_bytes), content, render_time
                    last_yield = t()
            # always render the final image of each iteration
            image_bytes, render_time = await render_html(
                page, apply_template(content, template)
            )
            history.append((content, image_bytes))
            yield to_pil(image_bytes), content, render_time
        # cleanup
        await browser.close()


demo = gr.Interface(
    generate_with_visual_feedback,
    inputs=[
        gr.Textbox(
            lines=5,
            label="Prompt",
            placeholder="Prompt to generate HTML",
            value="Generate a beautiful webpage for a cat cafe",
        ),
        gr.Dropdown(choices=["tailwind"], label="Template", value="tailwind"),
        gr.Dropdown(choices=["512", "1024"], label="Page Width", value="512"),
        gr.Slider(1, 10, 1, step=1, label="Iterations"),
    ],
    outputs=[
        gr.Image(type="pil", label="Rendered HTML", image_mode="RGB", format="png"),
        gr.Textbox(lines=5, label="Code"),
        gr.Number(label="Render Time", precision=2),
    ],
)


if __name__ == "__main__":
    demo.launch()
