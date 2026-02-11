import gradio as gr

from data import all_translations


def get_gradio_ui(gradio_use_function):
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
                    input_coef = gr.Slider(label='Коэфициент длительности', info="Коэфициент количества коротких нот",
                                           value=0.65,
                                           minimum=0.1,
                                           maximum=2.0, step=0.05)
                    input_tokens = gr.Number(label='Число токенов на выходе', value=1000, step=100)
                    generate_button = gr.Button("Сгенерировать шедевр", variant="primary")
                with gr.Column(scale=1):
                    out_text = gr.Textbox(label="Распознанный текст")
                    midi_out = gr.File(label='MIDI')
                    output_audio = gr.Audio(label="MP3")

            generate_button.click(
                fn=gradio_use_function,
                inputs=[input_prompt, input_temp, input_top_k, input_coef, input_tokens],
                outputs=[out_text, midi_out, output_audio]
            )

        with gr.Tab(label="Расширенный"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_prompt = gr.Text(label='Промпт')
                    input_temp = gr.Slider(label='Температура', value=0.8, minimum=0.1, maximum=2.0, step=0.01)
                    input_top_k = gr.Slider(label='Top-K',
                                            info="Число рассматриваемых вариантов ответа, меньше - меньше бреда и скучнее",
                                            value=50, minimum=5, maximum=200, step=5)
                    input_coef = gr.Slider(label='Коэфициент длительности', info="Коэфициент количества коротких нот",
                                           value=0.65,
                                           minimum=0.1,
                                           maximum=2.0, step=0.05)
                    input_tokens = gr.Number(label='Число токенов на выходе', value=1000, step=100)
                    generate_button = gr.Button("Сгенерировать шедевр", variant="primary")
                with gr.Column(scale=1):
                    out_text = gr.Textbox(label="Распознанный текст")
                    midi_out = gr.File(label='MIDI')
                    output_audio = gr.Audio(label="MP3")
            generate_button.click(
                fn=gradio_use_function,
                inputs=[input_prompt, input_temp, input_top_k, input_coef, input_tokens],
                outputs=[out_text, midi_out, output_audio]
            )
        with gr.Tab(label="Подражание"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_prompt = gr.Text(label='Промпт')
                    input_temp = gr.Slider(label='Температура', value=0.8, minimum=0.1, maximum=2.0, step=0.01)
                    input_top_k = gr.Slider(label='Top-K',
                                            info="Число рассматриваемых вариантов ответа, меньше - меньше бреда и скучнее",
                                            value=50, minimum=5, maximum=200, step=5)
                    input_coef = gr.Slider(label='Коэфициент длительности', info="Коэфициент количества коротких нот",
                                           value=0.65,
                                           minimum=0.1,
                                           maximum=2.0, step=0.05)
                    input_tokens = gr.Number(label='Число токенов на выходе', value=1000, step=100)
                    generate_button = gr.Button("Сгенерировать шедевр", variant="primary")
                with gr.Column(scale=1):
                    out_text = gr.Textbox(label="Распознанный текст")
                    midi_out = gr.File(label='MIDI')
                    output_audio = gr.Audio(label="MP3")
            generate_button.click(
                fn=gradio_use_function,
                inputs=[input_prompt, input_temp, input_top_k, input_coef, input_tokens],
                outputs=[out_text, midi_out, output_audio]
            )
        with gr.Tab(label="Алфавит"):
            gr.Json(value=all_translations)