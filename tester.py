import os
import source.document


# # Online PDF
# url = "https://css4.pub/2017/newsletter/drylab.pdf"
# doc = source.document.Document(url=url)
# content = doc.request_url()
# extracted_images = doc.extract_images(content)
# file_name = os.path.basename(url)
# output_folder = os.path.join("images", file_name)
# os.makedirs(output_folder, exist_ok=True)
# for i, image in enumerate(extracted_images):
#     file_path = os.path.join(output_folder, f"{i}.{image['ext']}")
#     with open(file_path, "wb") as f:
#         f.write(image["image_bytes"])
#     print(f"Saved {file_path}.")
    

# Offline PDF
url = "documents/Boeing.pdf"
doc = source.document.Document(url=url)
content = doc.read_local_file()
doc.content_type = "pdf"
extracted_images = doc.extract_images(content)
file_name = os.path.basename(url)
output_folder = os.path.join("images", file_name)
os.makedirs(output_folder, exist_ok=True)
for i, image in enumerate(extracted_images):
    file_path = os.path.join(output_folder, f"{i}.{image['ext']}")
    with open(file_path, "wb") as f:
        f.write(image["image_bytes"])
    print(f"Saved {file_path}.")