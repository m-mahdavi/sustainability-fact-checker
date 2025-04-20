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
        self.text_blocks = []
        self.images = []    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.embedding_model.eval()
        self.tokenizer = clip.tokenize
        self.bpe = clip.simple_tokenizer.SimpleTokenizer()

    def _read_pdf(self, path):
        with open(path, "rb") as f:
            return f.read()

    def extract_text_blocks(self, llm_model="llama3.2:3b-instruct-q4_K_M"):
        """Return a list of paragraphs with metadata across all reports."""
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
                            response: ChatResponse = chat(model=llm_model, messages=[
                                {"role": "user", "content": f"{prompt} {full_text}"},
                            ])
                            paragraphs = response["message"]["content"].split("<DELIMITER>")
                        new_paragraphs = []
                        for p in paragraphs: 
                            new_paragraphs += re.split(r"\n{2,}", p)
                        new_paragraphs = [p.strip() for p in new_paragraphs if p.strip()]
                        for text_block_id, paragraph in enumerate(new_paragraphs, start=1):
                            self.text_blocks.append({
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
        """Return a list of all images with metadata across all reports."""
        for report_path in self.reports:
            file_name = os.path.basename(report_path)
            try:
                pdf_bytes = self._read_pdf(report_path)
                with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                    for page_num, page in enumerate(doc, start=1):
                        for image_id, img in enumerate(page.get_images(full=True), start=1):
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_dir = os.path.join("../images", self.name, file_name.replace(".pdf", ""))
                            os.makedirs(image_dir, exist_ok=True)
                            image_path = os.path.join(image_dir, f"{image_id}.{base_image['ext']}")
                            with open(image_path, "wb") as img_file:
                                img_file.write(base_image["image"])
                            self.images.append({
                                "company": self.name,
                                "report": report_path,
                                "file": file_name,
                                "page": page_num,
                                "image_id": image_id,
                                "image_path": image_path,
                                "image_bytes": base_image["image"],
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
        """Embed a string of text or image bytes and return the embedding vector."""
        try:
            if isinstance(input_data, str):
                token_chunks = self._chunk_text(input_data).to(self.device)
                with torch.no_grad():
                    embeddings = self.embedding_model.encode_text(token_chunks)
                embeddings /= embeddings.norm(dim=-1, keepdim=True)
                embedding = embeddings.mean(dim=0)
                embedding /= embedding.norm()
                return embedding.cpu().numpy()

            elif isinstance(input_data, bytes):
                image = PIL.Image.open(io.BytesIO(input_data)).convert("RGB")
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embedding = self.embedding_model.encode_image(image_input)
                embedding /= embedding.norm(dim=-1, keepdim=True)
                return embedding.cpu().numpy()[0]

            else:
                raise ValueError("Input must be either a string (text) or bytes (image).")

        except Exception as e:
            print(f"Embedding error: {e}")
            return None
        
    def save_database(self, output_dir="../database"):
        """Save embedded text blocks and images to a JSON database."""
        os.makedirs(output_dir, exist_ok=True)
        all_records = []

        for block in self.text_blocks:
            block_copy = block.copy()
            record = {
                "id": str(uuid.uuid4()),
                "type": "text",
                "embedding": block_copy.pop("embedding").tolist(),
                "metadata": block_copy
            }
            all_records.append(record)

        for img in self.images:
            img_copy = img.copy()
            image_bytes = img_copy.pop("image_bytes")
            record = {
                "id": str(uuid.uuid4()),
                "type": "image",
                "embedding": img_copy.pop("embedding").tolist(),
                "metadata": img_copy
            }
            all_records.append(record)

        output_path = os.path.join(output_dir, f"{self.name}.json")
        with open(output_path, "w") as f:
            json.dump(all_records, f, indent=4)
        print(f"Saved {len(all_records)} records to {output_path}")

    def retrieve_evidence(self, query, k=10, database_path="../database"):
        """Retrieve the k most similar text and image records from the database."""
        try:
            query_embedding = self.embed(query)
            if query_embedding is None:
                print("Failed to embed the query.")
                return []

            file_path = os.path.join(database_path, f"{self.name}.json")
            with open(file_path, "r") as f:
                data = json.load(f)

            embeddings = np.array([item["embedding"] for item in data])
            similarities = sklearn.metrics.pairwise.cosine_similarity(query_embedding.reshape(1, -1), embeddings)
            similarities = similarities.flatten()
            top_indices = similarities.argsort()[-k:][::-1]
            top_similarities = similarities[top_indices]

            results = []
            for i in range(k):
                record = data[top_indices[i]]
                similarity_score = top_similarities[i]
                metadata = record["metadata"]
                if record["type"] == "text":
                    content = metadata.get("text", "No text content available.")
                else:
                    content = f"Image (ID: {metadata.get('image_id', 'Unknown')})"
                results.append({
                    "rank": i + 1,
                    "similarity": similarity_score,
                    "content": content,
                    "metadata": metadata
                })
            return results
        
        except Exception as e:
            print(f"Error during evidence retrieval: {e}")
            return []

    def verify_objective(self, objective, evidence, llm_model="llama3.2:3b-instruct-q4_K_M"):
        """Generate a verification report for a sustainability objective using evidence retrieved from the database."""
        try:
            evidence_text = "\n\n".join([f"Rank {i+1}: Similarity: {result['similarity']}\nContent: {result['content']}" for i, result in enumerate(evidence)])

            prompt = f"""
            Verify the following sustainability objective using the provided evidence:

            Sustainability Objective: "{objective}"

            Evidence:
            {evidence_text}

            Please generate a concise verification report that:
            - Assesses the truthfulness of the objective based on the provided evidence.
            - Describes any gaps or contradictions found in the evidence.
            - Provides any additional context or reasoning that could affect the credibility of the sustainability objective.
            """

            response: ChatResponse = chat(model=llm_model, messages=[{"role": "user", "content": prompt}])
            verification_report = response["message"]["content"]
            return verification_report

        except Exception as e:
            print(f"Error generating verification report: {e}")
            return None