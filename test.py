# from miditok import REMI, TokenizerConfig
# from symusic import Score
#
# # Creating a multitrack tokenizer, read the doc to explore all the parameters
# config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
# tokenizer = REMI(config)
#
# # Loads a midi, converts to tokens, and back to a MIDI
# midi = Score("midi/00000b8982c198eb5ba5540ac0aa6358.mid")
# tokens = tokenizer(midi)  # calling the tokenizer will automatically detect MIDIs, paths and tokens
# print(tokenizer.vocab)
#
# vocab = tokenizer.vocab
# print(dict(zip(vocab.values(), vocab.keys())))
# converted_back_midi = tokenizer(tokens)


import gradio as gr

def greet(prompt: str, temperature: float, top_k: float, duration: float, output_count: int):
    return str

demo = gr.Interface(
    fn=greet,
    inputs=[
        gr.Text(label='Промпт'),
        gr.Slider(label='Температура', value=0.83, minimum=0, maximum=2, step=0.01),
        gr.Slider(label='Top-K', info="Число рассматриваемых вариантов ответа, меньше - меньше бреда и скучнее", value=60, minimum=5, maximum=200, step=5),
        gr.Slider(label='Коэф. длительности', info="Коэфициент количества коротких нот", value=1.4, minimum=0.1, maximum=2.0, step=0.05),
        gr.Number(label='Число токенов на выходе', value=750)
    ],
    outputs=[gr.Audio(label="output")],
    allow_flagging="never",
    api_name="generate"
)

demo.launch()