import bs4
import fitz
import requests


class Document:
    
    def __init__(self, url, content_type=None, annotations=None):
        self.url = url
        self.content_type = content_type
        self.annotations = annotations
    
    def read_local_file(self):
        return open(self.url, "rb").read()
    
    def request_url(self):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.3"
        }
        response = requests.get(self.url, headers=headers, allow_redirects=True)
        response.raise_for_status()
        content = None
        if response.status_code == 200:
            content_type = response.headers.get("content-type", "")
            if "html" in content_type:
                self.content_type = "html" 
            elif "pdf" in content_type:
                self.content_type = "pdf"
            else:
                print(f"Unsupported content type for {self.url}: {content_type}")
                return None
            content = response.content
        return content
    
    def extract_images(self, content):
               
        def parse_html(html):
            extracted_images = []
            parsed_html = bs4.BeautifulSoup(html, "html.parser")
            removed_tags = ["style", "script"]
            for tag in parsed_html.find_all(removed_tags):
                tag.decompose()
            img_tags = parsed_html.find_all("img")
            for img_tag in img_tags:
                img_url = img_tag.get("src")
                if not img_url:
                    continue
                img_url = requests.compat.urljoin(self.url, img_url)
                try:
                    response = requests.get(img_url, stream=True)
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "")
                    if "image" in content_type:
                        image_bytes = response.content
                        image_ext = content_type.split("/")[-1]
                        extracted_images.append({
                            "image_bytes": image_bytes,
                            "ext": image_ext,
                            "url": img_url
                        })
                except requests.exceptions.RequestException as e:
                    print(f"Failed to download image from {img_url}: {e}")
            return extracted_images

        def parse_pdf(pdf_bytes):
            extracted_images = []
            with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf_document:
                for page_index in range(len(pdf_document)):
                    page = pdf_document.load_page(page_index)
                    images = page.get_images(full=True)
                    for img_index, img in enumerate(images):
                        xref = img[0] 
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        extracted_images.append({
                            "image_bytes": image_bytes,
                            "ext": image_ext,
                            "page_num": page_index + 1
                        })
            return extracted_images

        if self.content_type == "html":
            return parse_html(content)
        elif self.content_type == "pdf":
            return parse_pdf(content)
        else:
            return None
