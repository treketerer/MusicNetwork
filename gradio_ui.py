import gradio as gr
from gradio_xyslider import XYSlider
from functools import partial

from data.keywords import all_translations, prompt_words


def add_to_prompt(now_prompt: str, button_text: str):
    if len(now_prompt) != 0:
        return f"{now_prompt}, {button_text.lower()}"
    return button_text

def generate_prompt_tags(input_prompt):
    with gr.Accordion(label="💡Ключевые слова", open=False):
        for category in prompt_words.keys():
            with gr.Accordion(label=category, open=False):
                with gr.Row():
                    for tag in prompt_words[category]:
                        btn = gr.Button(value=tag)
                        btn.click(
                            fn=partial(add_to_prompt, button_text=tag),
                            inputs=[input_prompt],
                            outputs=input_prompt
                        )


def get_gradio_ui(inference_function, inference_imitation_function):
    gradio = gr.Blocks()
    with gradio:
        with gr.Tab(label="Промпт"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_prompt = gr.Text(label='Промпт')
                    input_temp = gr.Slider(label='Температура', value=0.8, minimum=0.1, maximum=2.0, step=0.01)
                    input_top_k = gr.Slider(label='Top-K',
                                            info="Число рассматриваемых вариантов ответа, меньше - меньше бреда и скучнее",
                                            value=50, minimum=5, maximum=200, step=5)
                    # input_coef = gr.Slider(label='Коэфициент длительности', info="Коэфициент количества коротких нот",
                    #                        value=0.65,
                    #                        minimum=0.1,
                    #                        maximum=2.0, step=0.05)
                    input_tokens = gr.Number(label='Число генерируемых тактов', value=10, step=1)
                    generate_button = gr.Button("Сгенерировать шедевр", variant="primary")
                    generate_prompt_tags(input_prompt)

                with gr.Column(scale=1):
                    out_text = gr.Textbox(label="Распознанный текст")
                    midi_out = gr.File(label='MIDI')
                    output_audio = gr.Audio(label="MP3")

            generate_button.click(
                fn=inference_function,
                inputs=[input_prompt, input_temp, input_top_k, input_tokens],
                outputs=[out_text, midi_out, output_audio]
            )

        with gr.Tab(label="Подражание"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_prompt = gr.Text(label='Промпт')
                    midi_input = gr.File(label='Поле для MIDI файла, с которого будет взят стиль', type="filepath",
                                         file_types=[".mid", ".midi"])
                    input_temp = gr.Slider(label='Температура', value=0.8, minimum=0.1, maximum=2.0, step=0.01)
                    input_top_k = gr.Slider(label='Top-K',
                                            info="Число рассматриваемых вариантов ответа, меньше - меньше бреда и скучнее",
                                            value=50, minimum=5, maximum=200, step=5)
                    input_tokens = gr.Number(label='Число генерируемых тактов', value=10, step=1)
                    generate_button = gr.Button("Сгенерировать шедевр", variant="primary")
                    generate_prompt_tags(input_prompt)

                with gr.Column(scale=1):
                    out_text = gr.Textbox(label="Распознанный текст")
                    midi_out = gr.File(label='MIDI')
                    output_audio = gr.Audio(label="MP3")
            generate_button.click(
                fn=inference_imitation_function,
                inputs=[input_prompt, input_temp, midi_input, input_top_k, input_tokens],
                outputs=[out_text, midi_out, output_audio]
            )
        with gr.Tab(label="Алфавит"):
            gr.Json(value=all_translations)

    return gradio