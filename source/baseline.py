import os
import re
import fitz
import uuid
import ollama

class Baseline:
    def __init__(self, name, base_dir="../documents"):
        self.name = name
        self.folder = os.path.join(base_dir, name)
        self.reports = sorted([
            os.path.join(self.folder, f)
            for f in os.listdir(self.folder)
            if f.lower().endswith(".pdf")
        ])
        self.text_blocks = []
        self.images = []

    def extract_text_and_images(self):
        for report_path in self.reports:
            file_name = os.path.basename(report_path)
            try:
                pdf_bytes = open(report_path, "rb").read()
                with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                    for page_num, page in enumerate(doc, start=1):
                        # Extract text
                        text = page.get_text("text").strip()
                        if text:
                            self.text_blocks.append({
                                "id": str(uuid.uuid4()),
                                "report": report_path,
                                "file": file_name,
                                "page": page_num,
                                "text": text
                            })

                        # Extract images
                        for image_id, img in enumerate(page.get_images(full=True), start=1):
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            if not base_image or "image" not in base_image:
                                continue
                            self.images.append({
                                "id": str(uuid.uuid4()),
                                "report": report_path,
                                "file": file_name,
                                "page": page_num,
                                "image_bytes": base_image["image"],
                                "ext": base_image["ext"],
                                "page_text": text
                            })
            except Exception as e:
                print(f"Failed to process {file_name}: {e}")

    def rank_images_for_query(self, query):
        query_terms = re.findall(r"\w+", query.lower())
        scored_images = []
        for img in self.images:
            text = img.get("page_text", "").lower()
            score = sum(text.count(term) for term in query_terms)
            scored_images.append((score, img))
        scored_images.sort(key=lambda x: x[0], reverse=True)
        ranked = [img for score, img in scored_images if score > 0]
        return ranked

    def verify_objective(self, objective, llm_model="llama3.2"):
        all_text = "\n\n".join([
            f"{tb['text']} [report: {tb['file']}, page: {tb['page']}]"
            for tb in self.text_blocks
        ])

        all_images = "\n".join([
            f"[Image from report: {img['file']}, page: {img['page']}, id: {img['id']}]"
            for img in self.images
        ])

        prompt = f"""Verify the following sustainability objective using ONLY the provided text evidence.

Objective: "{objective}"

Text Evidence:
{all_text}

Instructions:
You must strictly follow these formatting rules and output ONLY in the exact structure described below. 
Do NOT add, explain, summarize, restate the instructions, or include anything outside of the required output.

Task:
1. Write in Markdown format.
2. First line: State the final verdict — exactly one of: True, False, or Partially True.
3. Following lines: Present ONLY bullet points listing evidence that supports or contradicts the objective.
4. Each bullet must end with a reference in the exact format: **[report: xxx, page: yyy]**.
5. Do NOT include any section titles, extra text, or formatting beyond the verdict and bullet list."""

        response = ollama.chat(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "top_p": 1.0, "top_k": 0, "repeat_penalty": 1.0}
        )

        return response["message"]["content"].strip()
