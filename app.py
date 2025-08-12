import gradio as gr
from rag import answer_question

with gr.Blocks() as app:
    gr.Markdown("# Chat With Your Notes")
    gr.Markdown("Ask me anything based on your documents!")

    with gr.Row():
        chatbot = gr.Chatbot(label="Notes Assistant")

    with gr.Row():
        user_input = gr.Textbox(placeholder="Enter your question here...")
        send_button = gr.Button("Ask")

    def chat_response(message, history):
        history = history or []
        answer = answer_question(message)
        history.append((message, answer))
        return history, ""

    # when button is clicked
    send_button.click(chat_response, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    # when user presses enter
    user_input.submit(chat_response, inputs=[user_input, chatbot], outputs=[chatbot, user_input])

app.launch()
