from transformers import AutoTokenizer
import transformers
import torch
import argparse

class TransformersWrapper:
    def __init__(self, model_path="~/projects/def-zhu2048/cambish/llama3_1_8b_instruct"):
        self.model = transformers.pipeline(
            task="text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def generate(self, input, temperature, max_tokens, numSample):
        top_p = 0.95
        if numSample > 1:
            responses = []
            sequences = self.model(
                input,
                do_sample=True,
                top_k=1,
                num_return_sequences=numSample,
                max_new_tokens=max_tokens,
                return_full_text=False,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )

            for seq in sequences:
                response = seq['generated_text']
                responses.append(response)
            return responses

        else:
            sequences = self.model(
                input,
                do_sample=True,
                num_return_sequences=1,
                max_new_tokens=max_tokens,
                return_full_text=False,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )

            seq = sequences[0]
            response = seq['generated_text']

            return response



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Define Parameters")
    parser.add_argument('-test_dataset', action='store_true') # test custom prompt by default, set flag to run predictions over a specific dataset
    parser.add_argument("--temp", type=float, default=0.0, help = "temperature for sampling")
    parser.add_argument("--max_len", type=int, default=200, help = "max number of tokens in answer")
    parser.add_argument("--num_sample", type=int, default=1, help = "number of answers to sample")
    parser.add_argument("--model_path", type=str, default="~/projects/def-zhu2048/cambish/llama3_1_8b_instruct", help = "path to llm")
    args = parser.parse_args()

    # Set up Model
    model = TransformersWrapper(model_path=args.model_path)

    if not args.test_dataset:
        prompt = "What is the capital of Canada?"

        system_prompt = "You are a helpful assistant. Please answer the following question."

        input = {
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        }

        output = model.generate(input, args.temp, args.max_len, args.num_sample)

        print("The question being asked is: " + prompt)
        print("The generated answer is: \n" )
        print(output)