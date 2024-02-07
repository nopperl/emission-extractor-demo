from subprocess import run

import gradio as gr
from huggingface_hub import snapshot_download

from corporate_emission_reports.inference import extract_emissions

run(["sh", "install-llamacpp.sh"])
MODEL_PATH = snapshot_download("nopperl/emissions-extraction-lora-merged-GGUF")

def predict(input_method, document_file, document_url):
    document_path = document_file if input_method == "File" else document_url
    emissions = extract_emissions(document_path, MODEL_PATH, model_name="ggml-model-Q5_K_M.gguf")
    return emissions.model_dump_json()

with open("description.md", "r") as f:
    description = f.read().strip()

with open("article.md", "r") as f:
    article = f.read().strip()

interface = gr.Interface(
        predict,
        inputs=[gr.Radio(choices=["File", "URL"], value="File"), gr.File(type="filepath", file_types=[".pdf"], file_count="single", label="Report File"), gr.Textbox(label="Report URL")],
        outputs=gr.JSON(),
        description=description,
        examples = [
            ["URL", None, "https://www.deutsche-boerse.com/resource/blob/3373890/1cdeb942b1a02ce3495e25240dfdfe81/data/DBG-Detailed-GRI-index-Deutsche-Bo%CC%88rse-Group-AR-2022.pdf"],
            ["URL", None, "https://mpmaterials.com/downloads/MP_MATERIALS_2021_ESG_REPORT_A.pdf"],
        ],
        article=article,
        analytics_enabled=False,
        cache_examples=True,
    )
interface.queue().launch()

