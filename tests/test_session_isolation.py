import unittest
import uuid
from typing import Annotated, TypedDict
from unittest.mock import Mock, patch

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError

from src.reliability import InputValidationError
from src.session_store import get_session_store
from src.tools import vector_store
from src.tools.read_pdf import get_last_read_text, read_pdf, read_pdf_for_session
from src.tools.vector_store import (
    get_vectorstore,
    index_paper,
    index_paper_for_session,
    search_papers,
    search_papers_for_session,
)


class KeywordEmbeddings(Embeddings):
    """Small deterministic embedding model used only by offline tests."""

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(lowered.count("alpha")),
            float(lowered.count("beta")),
            1.0,
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class ToolState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    session_id: str


class SessionIsolationTests(unittest.TestCase):
    def setUp(self):
        self.store = get_session_store()
        self.store.clear()
        self.previous_embeddings = vector_store._embeddings
        vector_store._embeddings = KeywordEmbeddings()
        self.session_a = str(uuid.uuid4())
        self.session_b = str(uuid.uuid4())

    def tearDown(self):
        self.store.clear()
        vector_store._embeddings = self.previous_embeddings

    def test_last_read_pdf_is_isolated_by_session(self):
        response = Mock()
        response.headers = {}
        response.iter_content.side_effect = [[b"%PDF-a"], [b"%PDF-b"]]
        page_a = Mock()
        page_a.extract_text.return_value = "Alpha paper contents"
        page_b = Mock()
        page_b.extract_text.return_value = "Beta paper contents"

        with (
            patch("src.tools.read_pdf.request_with_retries", return_value=response),
            patch(
                "src.tools.read_pdf.PyPDF2.PdfReader",
                side_effect=[Mock(pages=[page_a]), Mock(pages=[page_b])],
            ),
        ):
            read_pdf_for_session("https://example.com/a.pdf", self.session_a)
            read_pdf_for_session("https://example.com/b.pdf", self.session_b)

        self.assertEqual(get_last_read_text(self.session_a), "Alpha paper contents")
        self.assertEqual(get_last_read_text(self.session_b), "Beta paper contents")

    def test_faiss_indexes_are_isolated_by_session(self):
        alpha_text = ("alpha attention mechanism and alpha representations. " * 30).strip()
        beta_text = ("beta optimization method and beta experiments. " * 30).strip()

        index_paper_for_session("Paper A", self.session_a, alpha_text)
        index_paper_for_session("Paper B", self.session_b, beta_text)

        result_a = search_papers_for_session("alpha", self.session_a)
        result_b = search_papers_for_session("beta", self.session_b)

        self.assertIn("Paper A", result_a)
        self.assertNotIn("Paper B", result_a)
        self.assertIn("Paper B", result_b)
        self.assertNotIn("Paper A", result_b)
        self.assertIsNot(
            get_vectorstore(self.session_a),
            get_vectorstore(self.session_b),
        )

    def test_embedding_clients_can_be_scoped_to_sessions(self):
        embeddings_a = KeywordEmbeddings()
        embeddings_b = KeywordEmbeddings()

        with patch(
            "src.tools.vector_store.GoogleGenerativeAIEmbeddings",
            side_effect=[embeddings_a, embeddings_b],
        ) as embedding_factory:
            vector_store.init_embeddings("secret-a", self.session_a)
            vector_store.init_embeddings("secret-b", self.session_b)

        resources_a = self.store.get(self.session_a)
        resources_b = self.store.get(self.session_b)
        self.assertIs(resources_a.embeddings, embeddings_a)
        self.assertIs(resources_b.embeddings, embeddings_b)
        self.assertIsNot(resources_a.embeddings, resources_b.embeddings)
        self.assertEqual(embedding_factory.call_count, 2)
        self.assertNotIn("secret-a", repr(resources_a))
        self.assertNotIn("secret-b", repr(resources_b))

    def test_session_id_is_hidden_from_llm_tool_schemas(self):
        self.assertNotIn("session_id", read_pdf.args)
        self.assertNotIn("session_id", index_paper.args)
        self.assertNotIn("session_id", search_papers.args)

    def test_vector_operations_validate_their_inputs(self):
        with self.assertRaises(InputValidationError):
            index_paper_for_session("   ", self.session_a, "valid paper text " * 20)
        with self.assertRaises(InputValidationError):
            search_papers_for_session("   ", self.session_a)
        with self.assertRaises(InputValidationError):
            search_papers_for_session("attention", self.session_a, k=0)

    def test_tool_node_injects_session_id_from_graph_state(self):
        alpha_text = ("alpha attention mechanism and representations. " * 30).strip()
        index_paper_for_session("Paper A", self.session_a, alpha_text)

        builder = StateGraph(ToolState)
        builder.add_node("tools", ToolNode([search_papers]))
        builder.add_edge(START, "tools")
        builder.add_edge("tools", END)
        graph = builder.compile()

        result = graph.invoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "search_papers",
                                "args": {"query": "alpha"},
                                "id": "search-call-1",
                                "type": "tool_call",
                            }
                        ],
                    )
                ],
                "session_id": self.session_a,
            }
        )

        self.assertIn("Paper A", result["messages"][-1].content)

    def test_api_rejects_invalid_session_ids(self):
        from src.api import ChatRequest

        with self.assertRaises(ValidationError):
            ChatRequest(message="hello", session_id="not-a-uuid")


if __name__ == "__main__":
    unittest.main()
