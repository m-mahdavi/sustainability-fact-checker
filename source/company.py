import io
import os
import re
import json
import uuid
import fitz
import gzip
import numpy as np
import sklearn.metrics.pairwise
import torch
import PIL.Image
import clip
import ollama


class Company:
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.embedding_model.eval()
        self.tokenizer = clip.tokenize
        self.bpe = clip.simple_tokenizer.SimpleTokenizer()

    def extract_text_blocks(self, llm_model="llama3.2:3b-instruct-q4_K_M"):
        for report_path in self.reports:
            file_name = os.path.basename(report_path)
            try:
                pdf_bytes = pdf_bytes = open(report_path, "rb").read()
                with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                    print(f"Processing company {self.name}, document {file_name}...")
                    for page_num, page in enumerate(doc, start=1):
                        full_text = page.get_text("text").strip()
                        paragraphs = []

                        if len(full_text) < 300:
                            paragraphs = [full_text]
                        else:
                            prompt = "The following text has been extracted from a page of a PDF. Segment this content into small text blocks according to the following rules: (1) Each block should represent a logical unit, such as a title, sentence, or short paragraph. (2) Each block must be meaningful, self-contained, and no longer than a short paragraph. (3) Each block must be shorter than 50 words. (4) If a block exceeds that length, divide it into smaller, logical blocks. Do not alter the content in any way. Only insert <DELIMITER> between logical text blocks. Do not add metadata, commentary, or any other formatting. The text is as follows:\n\n"                            
                            response = ollama.chat(model=llm_model, messages=[{"role": "user", "content": f"{prompt} {full_text}"}], options={"temperature": 0.0, "top_p": 1.0, "top_k": 0, "repeat_penalty": 1.0})
                            paragraphs = response["message"]["content"].split("<DELIMITER>")
                        new_paragraphs = []
                        for p in paragraphs:
                            new_paragraphs += re.split(r"\n{2,}", p)
                        new_paragraphs = [p.strip() for p in new_paragraphs if p.strip()]

                        for text_block_id, paragraph in enumerate(new_paragraphs, start=1):
                            self.text_blocks.append({
                                "id": str(uuid.uuid4()),
                                "type": "text",
                                "company": self.name,
                                "report": report_path,
                                "file": file_name,
                                "page": page_num,
                                "text_block_id": text_block_id,
                                "text": paragraph,
                                "embedding": self.embed(paragraph)
                            })
            except Exception as e:
                print(f"Failed to extract text from {file_name}: {e}")

    def extract_images(self):
        for report_path in self.reports:
            file_name = os.path.basename(report_path)
            try:
                pdf_bytes = open(report_path, "rb").read()
                with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                    for page_num, page in enumerate(doc, start=1):
                        for image_id, img in enumerate(page.get_images(full=True), start=1):
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            if not base_image or "image" not in base_image:
                                print(f"Warning: Failed to extract valid image from xref {xref} on page {page_num}")
                                continue

                            image_dir = os.path.join("../images", self.name, file_name.replace(".pdf", ""))
                            os.makedirs(image_dir, exist_ok=True)
                            image_path = os.path.join(image_dir, f"{image_id}.{base_image['ext']}")
                            with open(image_path, "wb") as img_file:
                                img_file.write(base_image["image"])

                            self.images.append({
                                "id": str(uuid.uuid4()),
                                "type": "image",
                                "company": self.name,
                                "report": report_path,
                                "file": file_name,
                                "page": page_num,
                                "image_id": image_id,
                                "image_path": image_path,
                                # "image_bytes": base_image["image"],
                                "ext": base_image["ext"],
                                "embedding": self.embed(base_image["image"])
                            })
            except Exception as e:
                print(f"Failed to extract images from {file_name}: {e}")

    def _chunk_text(self, text, max_len=77):
        tokens = self.bpe.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_len - 2):
            chunk = tokens[i:i + max_len - 2]
            chunk = [49406] + chunk + [49407]
            pad_len = max_len - len(chunk)
            chunk += [0] * pad_len
            chunks.append(torch.tensor(chunk))
        return torch.stack(chunks)

    def embed(self, input_data):
        try:
            if isinstance(input_data, str):
                token_chunks = self._chunk_text(input_data).to(self.device)
                with torch.no_grad():
                    embeddings = self.embedding_model.encode_text(token_chunks)
                embeddings /= embeddings.norm(dim=-1, keepdim=True)
                embedding = embeddings.mean(dim=0)
                embedding /= embedding.norm()
                return embedding.cpu().numpy().tolist()

            elif isinstance(input_data, bytes):
                image = PIL.Image.open(io.BytesIO(input_data)).convert("RGB")
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embedding = self.embedding_model.encode_image(image_input)
                embedding /= embedding.norm(dim=-1, keepdim=True)
                return embedding.cpu().numpy()[0].tolist()

            else:
                raise ValueError("Input must be either a string (text) or bytes (image).")

        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def save_database(self, output_dir="../database"):
        os.makedirs(output_dir, exist_ok=True)
        records = [item.copy() for item in self.text_blocks + self.images]
        output_path = os.path.join(output_dir, f"{self.name}.json.gz")
        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            json.dump(records, f)
        print(f"Saved {len(records)} compressed records to {output_path}")
        
    def load_database(self, input_dir="../database"):
        db_file = os.path.join(input_dir, f"{self.name}.json.gz")
        if os.path.exists(db_file):
            try:
                with gzip.open(db_file, "rt", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    if item.get("type") == "text":
                        self.text_blocks.append(item)
                    elif item.get("type") == "image":
                        self.images.append(item)
                print(f"Loaded {len(data)} compressed records from database: {db_file}")
            except Exception as e:
                print(f"Failed to load compressed database: {e}")

    def retrieve_evidence(self, query, k=10):
        try:
            query_embedding = self.embed(query)
            if query_embedding is None:
                print("Failed to embed the query.")
                return [], []

            query_embedding = np.array(query_embedding).reshape(1, -1)

            text_data = [item for item in self.text_blocks if isinstance(item.get("embedding"), list) and len(item["embedding"]) == 512]
            text_embeddings = np.array([np.array(item["embedding"]) for item in text_data])
            text_results = []
            if text_embeddings.size > 0:
                text_similarities = sklearn.metrics.pairwise.cosine_similarity(query_embedding, text_embeddings).flatten()
                top_text_indices = text_similarities.argsort()[-k:][::-1]
                for i in range(len(top_text_indices)):
                    record = text_data[top_text_indices[i]]
                    similarity_score = text_similarities[top_text_indices[i]]
                    text_results.append({
                        "rank": i + 1,
                        "similarity": float(similarity_score),
                        "record": record
                    })

            image_data = [item for item in self.images if isinstance(item.get("embedding"), list) and len(item["embedding"]) == 512]
            image_embeddings = np.array([np.array(item["embedding"]) for item in image_data])
            image_results = []
            if image_embeddings.size > 0:
                image_similarities = sklearn.metrics.pairwise.cosine_similarity(query_embedding, image_embeddings).flatten()
                top_image_indices = image_similarities.argsort()[-k:][::-1]
                for i in range(len(top_image_indices)):
                    record = image_data[top_image_indices[i]]
                    similarity_score = image_similarities[top_image_indices[i]]
                    image_results.append({
                        "rank": i + 1,
                        "similarity": float(similarity_score),
                        "record": record
                    })

            return text_results, image_results

        except Exception as e:
            print(f"Error during evidence retrieval: {e}")
            return [], []

    def verify_objective(self, objective, text_evidence, llm_model="llama3.2"):
        try:
            if not text_evidence:
                return "No text-based evidence found to verify the sustainability objective."

            new_text_evidence = "\n\n".join([
                f"{item['record']['text']} [report: {item['record']['file']}, page: {item['record']['page']}]"
                for item in text_evidence
            ])

            prompt = f"""Verify the following sustainability objective using ONLY the provided text evidence.

Objective: "{objective}"

Text Evidence:
{new_text_evidence}

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
            verification_report = response["message"]["content"].strip()
            return verification_report

        except Exception as e:
            print(f"Error generating verification report: {e}")
            return None

