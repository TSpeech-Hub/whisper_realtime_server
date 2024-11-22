from llama_cpp import Llama

context = "Ecco il testo completo fino a ora:"

fragments = [
    "This is the first part of the text", 
    "followed by another fragment.", 
    "Here comes a third piece to add", 
    "and then yet another one to extend", 
    "it further. The text continues with", 
    "more parts, each one fitting into", 
    "the whole. Let us keep going with", 
    "additional content! Making sure everything", 
    "stays in order is crucial.", 
    "Did we miss anything?", 
    "No, every part contributes to the", 
    "overall meaning so nothing should be overlooked", 
    "Now we add more words to the sequence", 
    "and build towards the conclusion...", 
    "The process requires precision.", 
    "Attention to detail is key!", 
    "Next, we include another piece ensuring smooth", 
    "transitions between sections.", 
    "Consistency is important for the final result", 
    "as we near the end of this long text", 
    "The text must make sense as a whole", 
    "and flow naturally without gaps. Careful handling", 
    "of each fragment matters because they all", 
    "belong together.", 
    "This is the penultimate piece of the puzzle...", 
    "and here comes the final one!"
]

llm = Llama.from_pretrained(
	repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
	filename="Phi-3-mini-4k-instruct-fp16.gguf",
)

def chat(frags):
    out = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant designed to strictly follow instructions and reassemble text fragments without adding or omitting any content. "
                    "Remember to add a new line whenever a line exceeds 80 characters. Your output should only contain the assembled text."
                )
            },
            {
                "role": "user",
                "content": 
                    f"Reassemble the following text fragments into the original whole. Do not add, remove, or modify anything. Ensure lines do not exceed 80 characters. "
                    f"Provide only the reassembled text:\n\n{frags}"
            }
        ]
    )
    return out['choices'][0]['message']['content']

def reattach(frags):
    for i in range(len(frags)//2):
        content = chat(frags[0:i+2])
        print(content)

reattach(fragments)
