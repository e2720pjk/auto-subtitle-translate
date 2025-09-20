from setuptools import setup, find_packages

setup(
    version="1.0.0",
    name="auto_subtitle_cli",
    packages=find_packages(),
    author="e2720pjk",
    author_email="e2720pjk@gmail.com",
    url="https://github.com/e2720pjk/auto-subtitle-translate",
    install_requires=[
        'openai-whisper',
        'ffmpeg-python',
        'transformers',
        'sentencepiece',
        'protobuf',
    ],
    description="Automatically generate, translate and embed subtitles into your videos",
    keywords=['subtitles', 'translate', 'video', 'whisper', 'llama2'],
    entry_points={
        'console_scripts': ['auto_subtitle_cli=auto_subtitle_cli.cli:cli_main'],
    },
    include_package_data=True,
)
