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
            ["URL", None, "https://www.bms.com/assets/bms/us/en-us/pdf/bmy-2022-esg-report.pdf"],
            ["URL", None, "https://www.7andi.com/library/dbps_data/_template_/_res/en/sustainability/sustainabilityreport/2022/pdf/2022_all_01.pdf"],
            ["URL", None, "https://www.infineon.com/dgdl/Sustainability_at+Infineon_2023.pdf?fileId=8ac78c8b8b657de2018c009d03120100"],
        ],
        article=article,
        analytics_enabled=False,
        cache_examples=False,
    )
interface.queue().launch()

