def load_documents(path:str) -> list[str]:
    with open(path, "r",encoding="utf-8") as f:
        text = f.read();
    documents = [line.strip() for line in text.split("\n") if line.strip()]
    return documents

