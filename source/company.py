import io
import os
import re
import json
import uuid
import fitz
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
                            response = ollama.chat(model=llm_model, messages=[{"role": "user", "content": f"{prompt} {full_text}"}])
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
        records = []

        for item in self.text_blocks + self.images:
            item_copy = item.copy()
            # item_copy.pop("image_bytes", None)
            records.append(item_copy)

        output_path = os.path.join(output_dir, f"{self.name}.json")
        with open(output_path, "w") as f:
            json.dump(records, f, indent=4)
        print(f"Saved {len(records)} records to {output_path}")
        
    def load_database(self, input_dir="../database"):
        db_file = os.path.join(input_dir, f"{self.name}.json")
        if os.path.exists(db_file):
            try:
                with open(db_file, "r") as f:
                    data = json.load(f)
                for item in data:
                    if item.get("type") == "text":
                        self.text_blocks.append(item)
                    elif item.get("type") == "image":
                        self.images.append(item)
                print(f"Loaded {len(data)} records from existing database: {db_file}")
            except Exception as e:
                print(f"Failed to load existing database: {e}")

    def retrieve_evidence(self, query, k=10, database_path="../database"):
        try:
            query_embedding = self.embed(query)
            if query_embedding is None:
                print("Failed to embed the query.")
                return []

            file_path = os.path.join(database_path, f"{self.name}.json")
            with open(file_path, "r") as f:
                data = json.load(f)

            data = [item for item in data if isinstance(item.get("embedding"), list) and len(item["embedding"]) == 512]
            embeddings = np.array([np.array(item["embedding"]) for item in data if "embedding" in item])
            if embeddings.size == 0:
                print("No embeddings found in the database.")
                return []

            query_embedding = np.array(query_embedding).reshape(1, -1)
            similarities = sklearn.metrics.pairwise.cosine_similarity(query_embedding, embeddings).flatten()
            top_indices = similarities.argsort()[-k:][::-1]
            top_similarities = similarities[top_indices]

            results = []
            for i in range(len(top_indices)):
                record = data[top_indices[i]]
                similarity_score = top_similarities[i]
                results.append({
                    "rank": i + 1,
                    "similarity": float(similarity_score),
                    "record": record
                })
            return results

        except Exception as e:
            print(f"Error during evidence retrieval: {e}")
            return []

    def verify_objective(self, objective, evidence, llm_model="llama3.2:3b-instruct-q4_K_M"):
        try:
            text_evidence = [
                item for item in evidence
                if item["record"].get("type") == "text" and "text" in item["record"]
            ]

            image_evidence = [
                item for item in evidence
                if item["record"].get("type") == "image"
            ]

            if not text_evidence:
                return "No text-based evidence found to verify the sustainability objective.", image_evidence

            evidence_text = "\n\n".join([f"{item['record']['text']}" for item in text_evidence])

            prompt = f"""
Verify the following sustainability objective using the provided text evidence.

Objective: "{objective}"

Text Evidence:
{evidence_text}

Please generate a concise verification report that:
- Starts with a final verdict on whether the objective is true, false, or partially true.
- Briefly lists all pieces of evidence that support or contradict the objective.
- Do not write anything else except the above-mentioned final verdict and bullet points of evidence in markdown.
"""

            response = ollama.chat(model=llm_model, messages=[{"role": "user", "content": prompt}])
            verification_report = response["message"]["content"].strip()
            return verification_report, image_evidence

        except Exception as e:
            print(f"Error generating verification report: {e}")
            return None, []
