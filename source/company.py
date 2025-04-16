import io
import os
import re
import fitz
from ollama import chat
from ollama import ChatResponse


class Company:

    def __init__(self, name, base_dir="../documents"):
        self.name = name
        self.folder = os.path.join(base_dir, name)
        self.reports = sorted([
            os.path.join(self.folder, f)
            for f in os.listdir(self.folder)
            if f.lower().endswith(".pdf")
        ])

    def _read_pdf(self, path):
        with open(path, "rb") as f:
            return f.read()

    def extract_text_blocks(self):
        """Return a list of paragraphs with metadata across all reports."""
        text_blocks = []
        for report_path in self.reports:
            file_name = os.path.basename(report_path)
            try:
                pdf_bytes = self._read_pdf(report_path)
                with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                    for page_num, page in enumerate(doc, start=1):
                        print(f"Processing company {self.name}, document {file_name}, page {page_num}...")
                        full_text = page.get_text("text").strip()
                        paragraphs = []
                        if len(full_text) < 300:
                            paragraphs = [full_text]
                        else:
                            prompt = "The following text has been extracted from a page of a PDF. Segment this content into small text blocks according to the following rules: (1) Each block should represent a logical unit, such as a title, sentence, or short paragraph. (2) Each block must be meaningful, self-contained, and no longer than a short paragraph. (3) Each block must be shorter than 50 words. (4) If a block exceeds that length, divide it into smaller, logical blocks. Do not alter the content in any way. Only insert <DELIMITER> between logical text blocks. Do not add metadata, commentary, or any other formatting. The text is as follows:\n\n"                            
                            response: ChatResponse = chat(model="llama3.2:3b-instruct-q4_K_M", messages=[
                                {
                                    "role": "user",
                                    "content": f"{prompt} {full_text}",
                                },
                            ])
                            paragraphs = response["message"]["content"].split("<DELIMITER>")
                        new_paragraphs = []
                        for p in paragraphs: 
                            new_paragraphs += re.split(r"\n{2,}", p)
                        new_paragraphs = [p.strip() for p in new_paragraphs if p.strip()]
                        for text_block_id, paragraph in enumerate(new_paragraphs, start=1):
                            text_blocks.append({
                                "company": self.name,
                                "report": report_path,
                                "file": file_name,
                                "page": page_num,
                                "text_block_id": text_block_id,
                                "text": paragraph
                            })
            except Exception as e:
                print(f"Failed to extract text from {file_name}: {e}")
        return text_blocks

    def extract_images(self):
        """Return a list of all images with metadata across all reports."""
        images = []
        for report_path in self.reports:
            file_name = os.path.basename(report_path)
            try:
                pdf_bytes = self._read_pdf(report_path)
                with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                    for page_num, page in enumerate(doc, start=1):
                        for image_id, img in enumerate(page.get_images(full=True), start=1):
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            images.append({
                                "company": self.name,
                                "report": report_path,
                                "file": file_name,
                                "page": page_num,
                                "image_id": image_id,
                                "image_bytes": base_image["image"],
                                "ext": base_image["ext"]
                            })
            except Exception as e:
                print(f"Failed to extract images from {file_name}: {e}")
        return images
