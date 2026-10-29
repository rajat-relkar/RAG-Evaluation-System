from pymupdf import open
import re
from typing import List
from pathlib import Path


def extract_paragraph(pdf_path: str | Path) -> List[str]:
    """This function takes a pdf path and returns an array of its paragraphs"""

    doc = open(pdf_path)
    paragraphs = []

    for page in doc:
        # get block from th page
        blocks = page.get_text("blocks")

        # sort blocks from top to bottom
        blocks = sorted(blocks, key=lambda b:(b[1], b[0]))

        for block in blocks:
            text = block[4]

            # Replace line breaks inside a paragraph with spaces
            text = re.sub(r"\r+", " ", text).strip()

            if text:
                paragraphs.append(text)

    return paragraphs

# print(extract_paragraph("../pdfs/Penguins_ACL.pdf"))
