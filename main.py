from agent.agent_initializer import create_agent
import argparse
import json
import time
from langchain.callbacks.base import BaseCallbackHandler
import tiktoken

class TokenAndToolCounter(BaseCallbackHandler):
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.tool_calls = 0
        try:
            self.enc = tiktoken.encoding_for_model(model_name)
        except Exception:
            self.enc = tiktoken.get_encoding("cl100k_base")

    def on_llm_start(self, serialized, prompts, **kwargs):
        for p in prompts:
            self.prompt_tokens += len(self.enc.encode(p))

    def on_llm_end(self, response, **kwargs):
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage") or {}
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
        elif hasattr(response, "generations"):
            for gen in response.generations:
                for m in gen:
                    self.completion_tokens += len(self.enc.encode(m.text))

    def on_tool_start(self, tool, input_str, **kwargs):
        self.tool_calls += 1

    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens


parser = argparse.ArgumentParser()
parser.add_argument("--oneshot", type=str, help="Run one prompt and exit")
parser.add_argument("--model-name", type=str, default="gpt-3.5-turbo", help="Model name for token counting")
args = parser.parse_args()

agent = create_agent()

if args.oneshot:
    t0 = time.time()
    metrics_cb = TokenAndToolCounter(model_name=args.model_name)
    try:
        if hasattr(agent, "invoke"):
            result = agent.invoke(args.oneshot, config={"callbacks": [metrics_cb]})
        elif hasattr(agent, "run"):
            result = agent.run(args.oneshot, callbacks=[metrics_cb])
        else:
            result = agent(args.oneshot)

        text = ""
        citations = []
        trace = {
            "retrieved": [],
            "latency": time.time() - t0,
            "tokens": metrics_cb.total_tokens(),
            "tool_calls": metrics_cb.tool_calls
        }

        if isinstance(result, dict):
            for k in ("result", "output", "answer", "output_text", "text"):
                if k in result and isinstance(result[k], str):
                    text = result[k]
                    break

            src_docs = result.get("source_documents", [])
            for d in src_docs:
                doc_id = d.metadata.get("doc_id") or d.metadata.get("source") or "doc"
                snippet = d.page_content[:500]
                citations.append({"doc_id": str(doc_id), "text": snippet})
                trace["retrieved"].append({"doc_id": str(doc_id)})

            if "intermediate_steps" in result and trace["tool_calls"] == 0:
                trace["tool_calls"] = len(result["intermediate_steps"])
        else:
            text = str(result)

        output = {
            "text": text,
            "citations": citations,
            "trace": trace
        }
        print(json.dumps(output, ensure_ascii=False))

    except Exception as e:
        print(json.dumps({
            "text": f"[error] {e}",
            "citations": [],
            "trace": {
                "retrieved": [],
                "latency": time.time() - t0,
                "tokens": metrics_cb.total_tokens(),
                "tool_calls": metrics_cb.tool_calls
            }
        }))
    exit(0)

if __name__ == "__main__":
    print("\n🤖 AI Agent with RAG is ready. Type 'exit' to quit.\n")
    while True:
        q = input("🧑 You: ")
        if q.lower() in ["exit", "quit"]:
            break
        try:
            result = agent.run(q)
            print(f"🤖 Agent: {result}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")
