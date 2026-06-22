from transformers import pipeline
import torch
 
path = "./weiser/101M-0.4"
prompt = "do i need to drop tyler? "

math_expressions = [
    "Ten plus five equals to ",
    "Eight minus three equals to ",
    "Six times two equals to ",
    "Nine plus seven equals to ",
    "Four times one equals to ",
    "Twelve minus six equals to ",
    "Three times five equals to ",
    "Seven plus zero equals to ",
    "Two times nine equals to ",
    "Eleven minus four equals to "
]

sentences = [
    "Cats love fish because of ",
    "Dogs chase balls because of ",
    "Birds build nests because of ",
    "Bears eat honey because of ",
    "Cows chew grass because of ",
    "Frogs catch flies because of ",
    "Bees make honey because of ",
    "Monkeys love bananas because of ",
    "Spiders spin webs because of ",
    "Horses eat hay because of "
]

sample_num = 3
system_promptr = "You are a calculator. You only output the numeric result of the given arithmetic expression. Never explain, never add words or symbols other than the number. Output exactly one line containing only the answer.\
\
Expression: 2 + 2\
Answer: 4\
\
Expression: 9 - 4\
Answer: 5\
\
Expression: 7 * 6\
Answer: 42\
\
Expression: 20 / 4\
Answer: 5\
\
Expression:" + math_expressions[sample_num] + "\
Answer:"

 
generator = pipeline("text-generation", model=path,max_new_tokens = 20, repetition_penalty=1.3, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="cpu")
print(generator(system_promptr)[0]['generated_text'])

