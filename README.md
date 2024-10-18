---
title: Html Agent
emoji: ⚡
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.1.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


# Updating packages.txt

When reading the build logs on HF, it says it's using the python:3.10 image.
https://hub.docker.com/layers/library/python/3.10/images/sha256-ad65dd4f9ada4720c144142187c3e88d7e346f6d78f94aa78853ebf76e1492c0?context=explore

This means we want the debian packages for debian 12, which we can find using a docker container:
```sh
docker run -it python:3.10 bash -c "pip install playwright && playwright install-deps --dry-run"
```

Now replace " " with "\n" in the output and write to packages.txt.
I tried doing this automatically, but the progress bar was interfering with the output, so I just did it manually.