import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from llava.conversation import conv_templates, SeparatorStyle

from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math




def eval_model(args):
    # Model
    model_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "flash_attention_2",
        "torch_dtype": "auto", 
    }


    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-vision-instruct", device_map="cuda", **model_kwargs)
    image_processor = AutoProcessor.from_pretrained("microsoft/Phi-3.5-vision-instruct", trust_remote_code=True)


    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")


    for line in tqdm(questions, total=len(questions)):

        messages = [
            {"role": "user", "content": "<|image_1|>\n" + line['text']},
        ]

        prompt = image_processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        image = Image.open(os.path.join(args.image_folder, line['image'])).convert('RGB')
        inputs = image_processor(prompt, [image], return_tensors="pt").to("cuda:0") 

        idx = line["question_id"]
        cur_prompt = line["text"]


        generate_ids = model.generate(
            **inputs,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            eos_token_id=[32007],
            max_new_tokens=128
            )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

        response = image_processor.batch_decode(generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False)[0] 


        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": response,
                                   "answer_id": ans_id,
                                   "model_id": 'phi3',
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="phi3")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
