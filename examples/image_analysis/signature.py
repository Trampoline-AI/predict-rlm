import dspy

from predict_rlm import File


class AnalyzeImages(dspy.Signature):
    """Analyze multiple images and answer the query about them.

    1. **List the image files** available in the input directory. Print
       their names and file sizes.

    2. **Load each image** as a base64 data URI using Python's base64 and
       pathlib modules.

    3. **Use predict()** with dspy.Image typed inputs to analyze the images
       in the context of the query. Process multiple images in parallel with
       asyncio.gather() if there are several.

    4. **Synthesize** the findings into a single answer that addresses the
       query across all images.
    """

    images: list[File] = dspy.InputField(desc="Image files to analyze (PNG, JPG, WEBP)")
    query: str = dspy.InputField(desc="A question about the images")
    answer: str = dspy.OutputField(desc="The answer to the query")
