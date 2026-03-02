import gradio as gr
from gradio_xyslider import XYSlider
from functools import partial

from data import all_translations, prompt_words


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
                    input_tokens = gr.Number(label='Число генерируемых тактов', value=10, step=1)
                    generate_prompt_tags(input_prompt)

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

                    lane = XYSlider(
                        label="Плоскость настроения",
                        x_min=-1.0,
                        x_max=1.0,
                        y_min=-1.0,
                        y_max=1.0,
                        upper_right_label="Весело/Быстро",
                        upper_left_label="Грустно/Быстро",
                        lower_right_label="Весело/Медленно",
                        lower_left_label="Грустно/Медленно",
                        color_upper_right="rgba(255, 59, 130, 0.85)", # red
                        color_upper_left="rgba(59, 130, 246, 0.85)",  # blue
                        color_lower_right="rgba(255, 165, 0, 0.85)",  # orange
                        color_lower_left="rgba(59, 255, 130, 0.85)",  # green
                    )

                    genre = gr.Dropdown([
                        "None", "Поп", "Рок", "Танцевальная", "Клубная", "Хаус", "Рэп", "Соул",
                        "Джаз", "Блюз", "Кантри", "Фолк", "Традиционная", "Классическая",
                        "Камерная", "Современная", "Прогрессивная", "Маршевая",
                        "Госпел", "Религиозная", "Рождественская", "Мировая"
                    ], label="Выбор жанра")
                    instr_groups = gr.Dropdown(['Клавиши', "Гитары", "Бас", "Струнные", "Хор", "Духовые", "Синты", "Амбиент эффекты", "Перкуссия", "Барабаны"], label="Выбор групп инструментов", info="Выбор общих групп инструментов, конкретные инструменты задаются через промпт", multiselect=True)

                    input_temp = gr.Slider(label='Температура', value=0.8, minimum=0.1, maximum=2.0, step=0.01)
                    input_top_k = gr.Slider(label='Top-K',
                                            info="Число рассматриваемых вариантов ответа, меньше - меньше бреда и скучнее",
                                            value=50, minimum=5, maximum=200, step=5)

                    input_tokens = gr.Number(label='Число генерируемых тактов', value=10, step=1)
                    generate_prompt_tags(input_prompt)

                    generate_button = gr.Button("Сгенерировать шедевр", variant="primary")
                with gr.Column(scale=1):
                    out_text = gr.Textbox(label="Распознанный текст")
                    midi_out = gr.File(label='MIDI')
                    output_audio = gr.Audio(label="MP3")
            generate_button.click(
                fn=gradio_use_function,
                inputs=[input_prompt, lane, genre, instr_groups, input_temp, input_top_k, input_tokens],
                outputs=[out_text, midi_out, output_audio]
            )
        with gr.Tab(label="Подражание"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_prompt = gr.Text(label='Промпт')
                    midi_input = gr.File(label='Поле для MIDI файла, с которого будет взят стиль')
                    input_temp = gr.Slider(label='Температура', value=0.8, minimum=0.1, maximum=2.0, step=0.01)
                    input_top_k = gr.Slider(label='Top-K',
                                            info="Число рассматриваемых вариантов ответа, меньше - меньше бреда и скучнее",
                                            value=50, minimum=5, maximum=200, step=5)
                    input_coef = gr.Slider(label='Коэфициент длительности', info="Коэфициент количества коротких нот",
                                           value=0.65,
                                           minimum=0.1,
                                           maximum=2.0, step=0.05)
                    input_tokens = gr.Number(label='Число генерируемых тактов', value=10, step=1)
                    generate_prompt_tags(input_prompt)

                    generate_button = gr.Button("Сгенерировать шедевр", variant="primary")
                with gr.Column(scale=1):
                    out_text = gr.Textbox(label="Распознанный текст")
                    midi_out = gr.File(label='MIDI')
                    output_audio = gr.Audio(label="MP3")
            generate_button.click(
                fn=gradio_use_function,
                inputs=[input_prompt, input_temp, midi_input, input_top_k, input_coef, input_tokens],
                outputs=[out_text, midi_out, output_audio]
            )
        with gr.Tab(label="Алфавит"):
            gr.Json(value=all_translations)

    return gradio