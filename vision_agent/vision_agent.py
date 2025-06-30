import argparse
import uuid
from functools import partial
from pathlib import Path

import gradio as gr
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import MessageGraph
from langgraph.checkpoint.memory import MemorySaver, PersistentDict


def create_graph(model: str = "llava", base_url: str = "http://127.0.0.1:11434"):
    llm = ChatOllama(model=model, base_url=base_url)

    def call_llm(messages):
        response = llm.invoke(messages)
        return [response]

    builder = MessageGraph()
    builder.add_node("llm", call_llm)
    builder.set_entry_point("llm")
    builder.set_finish_point("llm")

    memory = MemorySaver(factory=partial(PersistentDict, filename="chat.db"))
    return builder.compile(checkpointer=memory)


def build_interface(graph):
    thread_id = str(uuid.uuid4())

    def respond(message, image, history):
        content = [{"type": "text", "text": message}]
        display_user = message
        if image:
            file_path = Path(image).resolve()
            display_user += f"\n![image]({file_path.as_uri()})"
            content.append({"type": "image_url", "image_url": file_path.as_uri()})
        result = graph.invoke(
            [HumanMessage(content=content)],
            {"configurable": {"thread_id": thread_id}},
        )
        answer = result[-1].content
        history.append({"role": "user", "content": display_user})
        history.append({"role": "assistant", "content": answer})
        return history, "", None

    with gr.Blocks() as demo:
        gr.Markdown("# LangGraph Vision Chat")
        chatbot = gr.Chatbot(type="messages")
        with gr.Row():
            txt = gr.Textbox(placeholder="Enter message")
            img = gr.Image(type="filepath")
            send = gr.Button("Send")
        send.click(respond, [txt, img, chatbot], [chatbot, txt, img])
        txt.submit(respond, [txt, img, chatbot], [chatbot, txt, img])
    return demo


def main():
    parser = argparse.ArgumentParser(description="Run a vision-enabled agent")
    parser.add_argument("--model", default="llava", help="Ollama model name")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434", help="Ollama base URL")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    args = parser.parse_args()

    graph = create_graph(model=args.model, base_url=args.base_url)
    demo = build_interface(graph)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
