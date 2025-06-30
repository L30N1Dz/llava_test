"""Real-time object detection agent using a webcam feed."""

import argparse
import uuid
from functools import partial
from pathlib import Path

import cv2
import gradio as gr
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import MessageGraph
from langgraph.checkpoint.memory import MemorySaver, PersistentDict
from ultralytics import YOLO


def capture_and_detect(yolo, face_cascade):
    """Capture a single frame from the webcam and return annotated image path."""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read frame from webcam")

    annotated = frame.copy()
    results = yolo(frame)
    for r in results:
        for box in r.boxes:
            cls = yolo.names[int(box.cls[0])]
            if cls == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)

    out_path = Path("/tmp") / f"frame_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(out_path), annotated)
    return out_path


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


def build_interface(graph, yolo, face_cascade):
    thread_id = str(uuid.uuid4())

    def respond(message, history):
        img_path = capture_and_detect(yolo, face_cascade)
        content = [{"type": "text", "text": message}]
        display_user = message
        if img_path:
            file_path = Path(img_path).resolve()
            display_user += f"\n![image]({file_path.as_uri()})"
            content.append({"type": "image_url", "image_url": file_path.as_uri()})
        result = graph.invoke(
            [HumanMessage(content=content)],
            {"configurable": {"thread_id": thread_id}},
        )
        answer = result[-1].content
        history.append({"role": "user", "content": display_user})
        history.append({"role": "assistant", "content": answer})
        return history, ""

    with gr.Blocks() as demo:
        gr.Markdown("# Object Detector Agent")
        chatbot = gr.Chatbot(type="messages")
        txt = gr.Textbox(placeholder="Enter message")
        send = gr.Button("Send and Capture")
        send.click(respond, [txt, chatbot], [chatbot, txt])
        txt.submit(respond, [txt, chatbot], [chatbot, txt])
    return demo


def main():
    parser = argparse.ArgumentParser(description="Run a vision-enabled agent")
    parser.add_argument("--model", default="llava", help="Ollama model name")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434", help="Ollama base URL")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    args = parser.parse_args()

    yolo = YOLO("yolov8n.pt")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    graph = create_graph(model=args.model, base_url=args.base_url)
    demo = build_interface(graph, yolo, face_cascade)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
