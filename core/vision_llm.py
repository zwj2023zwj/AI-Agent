from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
from utils.logger import logger
import os

class VisionLLM:
    def __init__(self, model_id='microsoft/Florence-2-base', device=None):
        if device:
            self.device = device
        else:
            self.device = "cuda:2" if torch.cuda.is_available() else "cpu"
            
        logger.info(f"Loading Vision LLM ({model_id}) on {self.device}...")
        
        # Use trust_remote_code=True as Florence-2 requires it
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(self.device).eval()
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            logger.info("Vision LLM loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Vision LLM: {e}")
            raise e

    def run_task(self, image_path, task_prompt, text_input=None):
        """
        Generic function to run Florence-2 tasks.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Could not open image {image_path}: {e}")
            return "Error loading image"

        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=True,
            num_beams=3,
        )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )

        return parsed_answer

    def describe_image(self, image_path):
        """
        Generates a detailed caption.
        """
        # <MORE_DETAILED_CAPTION> is a specific task prompt for Florence-2
        result = self.run_task(image_path, "<MORE_DETAILED_CAPTION>")
        # print("ZWK:", result)
        return result.get("<MORE_DETAILED_CAPTION>", "No description generated.")

    # def visual_qa(self, image_path, question):
    #     """
    #     Answers a question about the image.
    #     """
    #     # <VQA> is the task prompt, we append the question
    #     # Note: Florence-2 format for VQA is usually: <VQA> Question
    #     # However, the post processor expects the task key.
    #     # Let's try generic generation for QA or use the specific VQA task
    #     result = self.run_task(image_path, "<VQA>", text_input=question)
        
    #     print(result)
    #     return result.get("<VQA>", "No answer generated.")
    
    def visual_qa(self, image_path, question):
        """
        Answers a question about the image with cleaner output.
        """
        # Adding a space often helps Florence-2 distinguish task from input
        result = self.run_task(image_path, "<VQA>", text_input=" " + question)
        answer = result.get("<VQA>", "No answer generated.")
        
        # Clean up the common "QA>question" prefix and potential tags
        if isinstance(answer, str):
            # Remove the echoed question if it's there
            if "QA>" in answer:
                answer = answer.split("QA>")[-1]
            if question in answer:
                answer = answer.replace(question, "").strip()
            
            # Remove any trailing/leading symbols or grounding tags if not wanted
            import re
            answer = re.sub(r'<[^>]+>', '', answer).strip()
            
            # If after cleaning we have nothing, return the original or a default
            if not answer:
                answer = "The model identified relevant regions but didn't provide a textual answer."
                
        return answer


# # from transformers import AutoModelForCausalLM, AutoTokenizer
# # from PIL import Image
# # import torch
# # from utils.logger import logger
# # import os
# # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# # class VisionLLM:
# #     def __init__(self, model_id='vikhyatk/moondream2', revision="2024-08-26", device=None):
# #         if device:
# #             self.device = device
# #         else:
# #             self.device = "cuda:2" if torch.cuda.is_available() else "cpu"
            
# #         logger.info(f"Loading Moondream LLM ({model_id}) on {self.device}...")
        
# #         try:
# #             # Moondream requires trust_remote_code=True and usually a specific revision for stability
# #             self.model = AutoModelForCausalLM.from_pretrained(
# #                 model_id, 
# #                 trust_remote_code=True, 
# #                 revision=revision
# #             ).to(self.device).eval()
# #             self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
# #             logger.info("Moondream LLM loaded successfully.")
# #         except Exception as e:
# #             logger.error(f"Failed to load Moondream LLM: {e}")
# #             # Fallback attempt without revision if specific one fails
# #             try:
# #                 logger.info("Attempting to load Moondream without revision...")
# #                 self.model = AutoModelForCausalLM.from_pretrained(
# #                     model_id, 
# #                     trust_remote_code=True
# #                 ).to(self.device).eval()
# #                 self.tokenizer = AutoTokenizer.from_pretrained(model_id)
# #                 logger.info("Moondream LLM loaded successfully (latest).")
# #             except Exception as e2:
# #                 logger.error(f"Critical failure loading Moondream: {e2}")
# #                 raise e2

# #     def describe_image(self, image_path):
# #         """
# #         Generates a caption using Moondream.
# #         """
# #         try:
# #             image = Image.open(image_path).convert("RGB")
# #             # Moondream has a specific method for captioning
# #             # We encode image first for efficiency
# #             enc_image = self.model.encode_image(image)
# #             description = self.model.answer_question(enc_image, "Describe this image in detail.", self.tokenizer)
# #             return description.strip()
# #         except Exception as e:
# #             logger.error(f"Error describing image with Moondream: {e}")
# #             return f"Error: {str(e)}"

# #     def visual_qa(self, image_path, question):
# #         """
# #         Answers a question about the image using Moondream.
# #         """
# #         try:
# #             image = Image.open(image_path).convert("RGB")
# #             enc_image = self.model.encode_image(image)
# #             answer = self.model.answer_question(enc_image, question, self.tokenizer)
# #             return answer.strip()
# #         except Exception as e:
# #             logger.error(f"Error in visual_qa with Moondream: {e}")
# #             return f"Error: {str(e)}"


# from transformers import AutoProcessor, AutoModelForVision2Seq
# try:
#     from qwen_vl_utils import process_vision_info
# except ImportError:
#     process_vision_info = None

# from PIL import Image
# import torch
# from utils.logger import logger
# import os

# class VisionLLM:
#     def __init__(self, model_id='Qwen/Qwen3-VL-2B-Instruct', device="cuda:2"):
#         if device:
#             self.device = device
#         else:
#             self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
#         logger.info(f"Loading Qwen3-VL LLM ({model_id}) on {self.device}...")
        
#         try:
#             # For Vision-Language models like Qwen2/3-VL, use AutoModelForVision2Seq
#             # Use torch_dtype instead of dtype as it's the standard argument for from_pretrained
#             # The warning might be coming from the model's internal config, but from_pretrained expects torch_dtype
#             self.model = AutoModelForVision2Seq.from_pretrained(
#                 model_id, 
#                 trust_remote_code=True, 
#                 torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
#                 # device_map="auto" if self.device == "cuda" else None
#             )
#             self.model.to(self.device).eval()
            
#             self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
#             logger.info("Qwen3-VL LLM loaded successfully.")
#         except Exception as e:
#             logger.error(f"Failed to load Qwen3-VL LLM: {e}")
#             raise e

#     def _generate(self, image_path, prompt):
#         """
#         Internal helper for Qwen-VL generation logic.
#         """
#         if process_vision_info is None:
#             return "Error: qwen-vl-utils not installed."

#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
#                     {"type": "text", "text": prompt},
#                 ],
#             }
#         ]

#         # Preparation for inference
#         text = self.processor.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
#         image_inputs, video_inputs = process_vision_info(messages)
#         inputs = self.processor(
#             text=[text],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",
#         )
#         inputs = inputs.to(self.device)

#         # Inference: Generation of the output
#         generated_ids = self.model.generate(**inputs, max_new_tokens=512)
#         generated_ids_trimmed = [
#             out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#         ]
#         output_text = self.processor.batch_decode(
#             generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#         )
#         return output_text[0]

#     def describe_image(self, image_path):
#         """
#         Generates a detailed caption using Qwen3-VL.
#         """
#         try:
#             return self._generate(image_path, "Describe this image in detail.")
#         except Exception as e:
#             logger.error(f"Error describing image with Qwen3-VL: {e}")
#             return f"Error: {str(e)}"

#     def visual_qa(self, image_path, question):
#         """
#         Answers a question about the image using Qwen3-VL.
#         """
#         try:
#             return self._generate(image_path, question)
#         except Exception as e:
#             logger.error(f"Error in visual_qa with Qwen3-VL: {e}")
#             return f"Error: {str(e)}"


# from transformers import BlipProcessor, BlipForQuestionAnswering
# from PIL import Image
# import torch
# from utils.logger import logger
# import os

# class VisionLLM:
#     def __init__(self, model_id='Salesforce/blip-vqa-base', device="cuda:2"):
#         if device:
#             self.device = device
#         else:
#             self.device = "cuda:2" if torch.cuda.is_available() else "cpu"
            
#         logger.info(f"Loading BLIP LLM ({model_id}) on {self.device}...")
        
#         try:
#             # BLIP for VQA uses BlipForQuestionAnswering
#             self.model = BlipForQuestionAnswering.from_pretrained(model_id).to(self.device).eval()
#             self.processor = BlipProcessor.from_pretrained(model_id)
#             logger.info("BLIP LLM loaded successfully.")
#         except Exception as e:
#             logger.error(f"Failed to load BLIP LLM: {e}")
#             raise e

#     def describe_image(self, image_path):
#         """
#         Generates a caption using BLIP. 
#         Note: blip-vqa-base is optimized for QA, but can do basic captioning.
#         """
#         try:
#             image = Image.open(image_path).convert("RGB")
#             # For blip-vqa-base, it's best to provide a prompt like "a picture of" or similar
#             # for "pseudo-captioning" if using the VQA model.
#             text = "a picture of"
#             inputs = self.processor(image, text, return_tensors="pt").to(self.device)
            
#             with torch.no_grad():
#                 # Note: BlipForQuestionAnswering doesn't have a .generate() in the same way 
#                 # as ConditionalGeneration models for pure captioning, but we can use 
#                 # the VQA aspect with a generic prompt.
#                 out = self.model.generate(**inputs)
#                 description = self.processor.decode(out[0], skip_special_tokens=True)
#             return description.strip()
#         except Exception as e:
#             logger.error(f"Error describing image with BLIP: {e}")
#             return f"Error: {str(e)}"

#     def visual_qa(self, image_path, question):
#         """
#         Answers a question about the image using BLIP.
#         """
#         try:
#             image = Image.open(image_path).convert("RGB")
#             inputs = self.processor(image, question, return_tensors="pt").to(self.device)
            
#             with torch.no_grad():
#                 out = self.model.generate(**inputs)
#                 answer = self.processor.decode(out[0], skip_special_tokens=True)
#             return answer.strip()
#         except Exception as e:
#             logger.error(f"Error in visual_qa with BLIP: {e}")
#             return f"Error: {str(e)}"