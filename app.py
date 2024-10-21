import gradio as gr
from playwright.async_api import async_playwright, Page
from PIL import Image
from io import BytesIO
from anthropic import Anthropic
from dotenv import load_dotenv
import os
from typing import Literal
import time
from base64 import b64encode
from contextlib import asynccontextmanager

load_dotenv()
# check for ANTHROPIC_API_KEY
if os.getenv("ANTHROPIC_API_KEY") is None:
    raise ValueError(
        "ANTHROPIC_API_KEY is not set, set it in .env or export it in your environment"
    )

anthropic = Anthropic()
model = "claude-3-5-sonnet-20240620"


def prepare_playwright_if_needed():
    # hopefully, we already installed the deps with dependencies.txt
    os.system("playwright install chromium")


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


def messages_text_to_web(prompt):
    return [
        {"role": "user", "content": prompt},
    ]


# returns the full text of the response each time
def stream_claude(messages, system=system_prompt, max_tokens=2000):
    text = ""
    with anthropic.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    ) as stream:
        for chunk in stream.text_stream:
            text += chunk
            yield text


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


def visual_feedback_messages(prompt, history: list[tuple[str, bytes]]):
    """
    history is a list of tuples of (content, image) corresponding to iterations of generation and rendering
    """
    improve_prompt = """
    Given the current draft of the webpage you generated for me as HTML and the screenshot of it rendered, improve the HTML to look nicer.
    """
    return [
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


def match_image_messages(image_bytes: bytes, history: list[tuple[bytes, bytes]]):
    improve_prompt = """
    Given the current draft of the webpage you generated for me as HTML and the original screenshot, improve the HTML to match closer to the original screenshot.
    """

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please generate a webpage that matches the image below as closely as possible:",
                },
                format_image(image_bytes, "image/png"),
            ],
        },
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


async def render_html(page: Page, content: str):
    start_time = time.perf_counter()
    await page.set_content(content)
    # weird, can we set scale to 2.0 directly instead of "device", ie whatever server this is running on?
    image_bytes = await page.screenshot(type="png", scale="device", full_page=True)
    dt = time.perf_counter() - start_time
    return image_bytes, dt


def apply_template(content, template):
    if template == "tailwind":
        return apply_tailwind(content)
    return content


def to_pil(image_bytes: bytes):
    return Image.open(BytesIO(image_bytes))


@asynccontextmanager
async def browser(width, height):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": width, "height": height})
        try:
            yield page
        finally:
            await browser.close()


async def throttle(generator, every=0.25):
    last_emit_time = 0
    for item in generator:
        current_time = time.perf_counter()
        if current_time - last_emit_time >= every:
            yield item
            last_emit_time = current_time
    # always emit the last item
    yield item


async def generate_with_visual_feedback(
    prompt,
    template,
    resolution: str = "512",
    num_iterations: int = 1,
):
    width = {"512": 512, "1024": 1024}[resolution]
    async with browser(width, width) as page:
        history = []
        for i in range(num_iterations):
            messages = (
                messages_text_to_web(prompt)
                if i == 0
                else visual_feedback_messages(prompt, history)
            )
            content = ""
            async for content in throttle(stream_claude(messages), every=0.25):
                image_bytes, render_time = await render_html(
                    page, apply_template(content, template)
                )
                yield to_pil(image_bytes), content, render_time
            history.append((content, image_bytes))


def to_image_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


async def match_image_with_visual_feedback(image, template, resolution, num_iterations):
    width = {"512": 512, "1024": 1024}[resolution]
    async with browser(width, width) as page:
        history = []
        for i in range(num_iterations):
            image.thumbnail((width, width), Image.Resampling.LANCZOS)
            messages = match_image_messages(to_image_bytes(image), history)
            async for content in throttle(stream_claude(messages), 0.25):
                image_bytes, render_time = await render_html(
                    page, apply_template(content, template)
                )
                yield to_pil(image_bytes), content, render_time
            # always render the final image of each iteration
            image_bytes, render_time = await render_html(
                page, apply_template(content, template)
            )
            history.append((content, image_bytes))


demo_generate = gr.Interface(
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

demo_match_image = gr.Interface(
    match_image_with_visual_feedback,
    inputs=[
        gr.Image(type="pil", label="Original Image", image_mode="RGB", format="png"),
        gr.Dropdown(choices=["tailwind"], label="Template", value="tailwind"),
        gr.Dropdown(choices=["512", "1024"], label="Page Width", value="512"),
        gr.Slider(1, 10, 3, step=1, label="Iterations"),
    ],
    outputs=[
        gr.Image(type="pil", label="Rendered HTML", image_mode="RGB", format="png"),
        gr.Textbox(lines=5, label="Code"),
        gr.Number(label="Render Time", precision=2),
    ],
)

demo = gr.TabbedInterface(
    [demo_match_image, demo_generate],
    ["Match Image", "Generate"],
)


if __name__ == "__main__":
    prepare_playwright_if_needed()
    demo.launch()
